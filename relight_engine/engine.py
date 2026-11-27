from __future__ import annotations

from collections import OrderedDict
from copy import copy
from dataclasses import replace
import hashlib
import os
import threading

import numpy as np

from .contracts import (
    CameraModel,
    SceneBuffers,
    LightingRig,
    EngineSettings,
    RelightResult,
)
from .geometry import (
    prepare_geometry,
    refine_depth_to_rgb_edges,
    clean_depth_background_skirt,
    integrate_normals_with_depth_prior,
    extract_micro_normal_bump,
)
from .lighting import evaluate_rig, compute_indirect_bounce, compute_indirect_specular, compute_volumetric_beam
from .calibration import (
    suggest_material_priors_from_semantics,
    refine_source_shading_with_albedo,
    calibrate_conductor_from_appearance,
    get_skin_weight_for_scene,
)
from .composite import (
    compose_hybrid_neural,
    compose_intrinsic,
    compose_source_preserving,
    _compose_intrinsic_output,
    _depth_guided_smooth,
    _get_or_compute_intrinsic_priors,
    _get_or_compute_intrinsic_residual,
)
from .preview import downsample_scene_buffers_for_preview, depth_guided_upsample_batch
from .quality import evaluate_scene_buffers, describe_material_response, evaluate_preservation
from .color import luminance


def _pipeline_observability(
    scene: SceneBuffers,
    requested_mode: str,
    effective_mode: str,
    target_rig: LightingRig,
) -> dict:
    """Explicit observability of effective mode, fallback reason, and active shading paths."""
    if requested_mode == "auto" and effective_mode == "ratio":
        fallback_reason = "intrinsic_albedo_unavailable_using_ratio_fallback"
    elif requested_mode not in ("auto", effective_mode):
        fallback_reason = f"switched_from_{requested_mode}_to_{effective_mode}"
    else:
        fallback_reason = None

    has_authored_mat = any(
        getattr(scene, k) is not None
        for k in ("roughness", "metalness", "specular_level", "specular_color", "clearcoat", "subsurface", "anisotropy")
    )
    has_model_mat = bool(scene.metadata.get("material_maps_estimated", False))
    env = target_rig.environment
    env_active = bool(
        env is not None
        and (
            float(env.strength) > 0.0
            or float(env.key_strength) > 0.0
            or (env.env_map is not None and float(env.env_map_strength) > 0.0)
        )
    )
    if has_model_mat and has_authored_mat:
        spec_path = "model_material_ggx"
    elif has_authored_mat:
        spec_path = "authored_material_ggx"
    else:
        spec_path = "conservative_prior_dielectric_ggx"
    if env_active:
        spec_path = f"{spec_path}+environment_ibl"

    return {
        "requested_renderer_mode": requested_mode,
        "fallback_reason": fallback_reason,
        "scene_buffers_available": {
            "albedo": scene.albedo is not None,
            "source_shading": scene.source_shading is not None,
            "non_diffuse_residual": scene.non_diffuse_residual is not None,
            "normals_supplied": scene.normals is not None,
            "material_maps_supplied": has_authored_mat,
        },
        "active_specular_path": spec_path,
        "residual_decomposition_active": bool(
            effective_mode == "intrinsic" and scene.non_diffuse_residual is not None
        ),
        "environment_lighting_active": env_active,
    }

_PREVIEW_SCENE_CACHE: OrderedDict[tuple, tuple[SceneBuffers, CameraModel]] = OrderedDict()
_PREVIEW_CACHE_LOCK = threading.Lock()
_PREVIEW_CACHE_MAX = 4


def _buf_digest(arr: np.ndarray | None) -> tuple | None:
    if arr is None:
        return None
    buf = np.ascontiguousarray(arr)
    return (buf.shape, str(buf.dtype), hashlib.blake2b(memoryview(buf), digest_size=16).digest())


def _get_or_create_preview_scene(
    scene: SceneBuffers,
    camera: CameraModel,
    max_side: int,
) -> tuple[SceneBuffers, CameraModel]:
    sem_digests = tuple((k, _buf_digest(scene.semantic_masks[k])) for k in sorted(scene.semantic_masks))
    inst_digests = tuple((k, _buf_digest(scene.instance_masks[k])) for k in sorted(scene.instance_masks))
    raw_digests = tuple((k, _buf_digest(scene.raw_material_maps[k])) for k in sorted(scene.raw_material_maps))
    key = (
        _buf_digest(scene.rgb_srgb),
        _buf_digest(scene.depth),
        _buf_digest(scene.depth_confidence),
        _buf_digest(scene.geometry_confidence),
        _buf_digest(scene.normals),
        _buf_digest(scene.albedo),
        _buf_digest(scene.source_shading),
        _buf_digest(scene.non_diffuse_residual),
        _buf_digest(scene.lightable_mask),
        _buf_digest(scene.protected_mask),
        _buf_digest(scene.alpha_mask),
        _buf_digest(scene.roughness),
        _buf_digest(scene.metalness),
        _buf_digest(scene.specular_level),
        _buf_digest(scene.specular_color),
        _buf_digest(scene.clearcoat),
        _buf_digest(scene.clearcoat_roughness),
        _buf_digest(scene.subsurface),
        _buf_digest(scene.anisotropy),
        _buf_digest(scene.anisotropy_angle),
        _buf_digest(scene.material_confidence),
        sem_digests,
        inst_digests,
        raw_digests,
        scene.depth_kind,
        bool(scene.material_metadata.get("confidence_applied", False)),
        int(max_side),
        float(camera.fx),
        float(camera.fy),
        float(camera.cx),
        float(camera.cy),
    )
    with _PREVIEW_CACHE_LOCK:
        cached = _PREVIEW_SCENE_CACHE.get(key)
        if cached is not None:
            _PREVIEW_SCENE_CACHE.move_to_end(key)
            return cached
    prev_scene, prev_cam = downsample_scene_buffers_for_preview(scene, camera, int(max_side))
    with _PREVIEW_CACHE_LOCK:
        _PREVIEW_SCENE_CACHE[key] = (prev_scene, prev_cam)
        _PREVIEW_SCENE_CACHE.move_to_end(key)
        while len(_PREVIEW_SCENE_CACHE) > _PREVIEW_CACHE_MAX:
            _PREVIEW_SCENE_CACHE.popitem(last=False)
    return prev_scene, prev_cam


def _fit_sh_source_irradiance(
    scene: SceneBuffers,
    normals: np.ndarray,
    source_d: np.ndarray,
) -> np.ndarray:
    """Fit an order-2 RGB Spherical Harmonics source light model from (normals, source_shading)."""
    assert scene.source_shading is not None
    sh = np.asarray(scene.source_shading, dtype=np.float32)
    if sh.ndim == 2:
        sh = np.repeat(sh[..., None], 3, axis=-1)
    active = (scene.depth > 0) & (scene.lightable_mask > 0.05)
    if scene.protected_mask is not None:
        active = active & (scene.protected_mask < 0.95)
    if not np.any(active):
        active = scene.depth > 0

    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    basis = np.stack([
        np.ones_like(nx),
        ny,
        nz,
        nx,
        nx * ny,
        ny * nz,
        3.0 * nz * nz - 1.0,
        nx * nz,
        nx * nx - ny * ny,
    ], axis=-1).astype(np.float64)

    sh_y = luminance(sh)
    # Exclude dark-albedo/low-shading pixels so dark clothing or hair doesn't pull SH toward zero.
    med_all = max(float(np.median(sh_y[active])), 1e-4)
    mask_fit = active & (sh_y > max(0.18, 0.35 * med_all))
    if np.sum(mask_fit) < 16:
        mask_fit = active
    b_fit = basis[mask_fit]
    y_fit = sh[mask_fit].astype(np.float64)
    if b_fit.shape[0] > 20000:
        step = max(1, b_fit.shape[0] // 20000)
        b_fit = b_fit[::step]
        y_fit = y_fit[::step]

    # Stronger L2 regularization on order-1/2 harmonics prevents extrapolation dark-holes on minority normals.
    reg_diag = np.array([0.02, 0.25, 0.25, 0.25, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=np.float64)
    gram = b_fit.T @ b_fit + np.diag(reg_diag * max(b_fit.shape[0] * 1e-3, 1.0))
    rhs = b_fit.T @ y_fit
    coeffs = np.linalg.solve(gram, rhs)
    sh_pred = np.maximum(basis @ coeffs, 1e-4).astype(np.float32)

    pred_y = np.maximum(luminance(sh_pred), 1e-4)
    sd_y = np.maximum(luminance(source_d), 1e-4)
    med_pred = max(float(np.median(pred_y[active])), 1e-4)
    med_sd = max(float(np.median(sd_y[active])), 1e-4)
    # Physical ambient floor (0.45x median) prevents blowing out eye sockets, brows, or under-chins.
    scaled = sh_pred * (med_sd / med_pred)
    return np.clip(scaled, 0.45 * med_sd, 3.0 * med_sd).astype(np.float32)


def _rigs_identical(a: LightingRig, b: LightingRig) -> bool:
    if a is b:
        return True
    if a != b:
        return False
    ea, eb = a.environment, b.environment
    if ea is None and eb is None:
        return True
    assert ea is not None and eb is not None
    if ea.env_map is eb.env_map:
        return True
    if ea.env_map is None or eb.env_map is None:
        return False
    return ea.env_map.shape == eb.env_map.shape and bool(np.array_equal(ea.env_map, eb.env_map))


def _effective_source_diffuse(
    scene: SceneBuffers,
    normals: np.ndarray,
    source_d: np.ndarray,
    settings: EngineSettings,
) -> np.ndarray:
    """Combine analytic source_rig with optional order-2 SH fit and depth-smoothed source shading."""
    base_src = source_d
    if settings.auto_sh_source and scene.source_shading is not None:
        base_src = _fit_sh_source_irradiance(scene, normals, source_d)

    # Guard against collapsed source estimates on illustrations/low-confidence scenes without albedo:
    # anchor implied source illumination cleanly so bright midtones do not blow out at the ratio ceiling.
    if scene.albedo is None and scene.source_shading is None:
        valid = (scene.depth > 0) & (scene.lightable_mask > 0.05)
        if np.any(valid):
            sd_y = np.maximum(luminance(base_src), 1e-4)
            med_sd = float(np.median(sd_y[valid]))
            if med_sd < 0.40:
                from .color import srgb_to_linear
                src_y = np.maximum(luminance(srgb_to_linear(scene.rgb_srgb)), 1e-4)
                smooth_y = np.clip(_depth_guided_smooth(src_y, scene.depth, 3.0), 0.30, 1.25)
                floor_w = float(np.clip((0.40 - med_sd) / 0.25, 0.0, 1.0))
                base_src = np.maximum(base_src, (smooth_y * floor_w)[..., None]).astype(np.float32)

    blend = float(settings.source_shading_blend)
    if blend <= 0.0 or scene.source_shading is None:
        return base_src
    sh = np.asarray(scene.source_shading, dtype=np.float32)
    if sh.ndim == 2:
        sh = np.repeat(sh[..., None], 3, axis=-1)
    smooth_sigma = max(float(settings.lighting_smooth_sigma) * 2.0, 2.0)
    sh_smooth = np.maximum(_depth_guided_smooth(sh, scene.depth, smooth_sigma), 1e-4)
    active = (scene.depth > 0) & (scene.lightable_mask > 0.05)
    if scene.protected_mask is not None:
        active = active & (scene.protected_mask < 0.95)
    if not np.any(active):
        active = scene.depth > 0
    sh_y = np.maximum(luminance(sh_smooth), 1e-4)
    sd_y = np.maximum(luminance(base_src), 1e-4)
    med_sh = max(float(np.median(sh_y[active])), 1e-4)
    med_sd = max(float(np.median(sd_y[active])), 1e-4)
    rel_y = np.clip(sh_y / med_sh, 0.42, 2.4)
    if os.environ.get("RELIGHT_ABLATE_EFF_SD_FLOOR") == "1":
        # Phase 1 ablation hook: floor source_shading to geometric SH irradiance across all pixels.
        rel_y = np.maximum(rel_y, 0.90 * (sd_y / med_sd))
    skin_w = get_skin_weight_for_scene(scene, settings)
    if skin_w is not None:
        # Stage 1 (Source floor): prevent skin albedo bias from collapsing eff_sd on skin_w pixels.
        floor_k = float(settings.skin_fidelity.floor_k)
        rel_y = (1.0 - skin_w) * rel_y + skin_w * np.maximum(rel_y, floor_k * (sd_y / med_sd))
    chroma = np.clip(sh_smooth / sh_y[..., None], 0.45, 2.0)
    chroma = np.power(chroma, 0.70)
    sh_matched = (rel_y[..., None] * chroma) * med_sd
    eff_out = ((1.0 - blend) * base_src + blend * sh_matched).astype(np.float32)
    if skin_w is not None:
        floor_k = float(settings.skin_fidelity.floor_k)
        eff_out = ((1.0 - skin_w[..., None]) * eff_out + skin_w[..., None] * np.maximum(eff_out, floor_k * source_d)).astype(np.float32)
    return eff_out


def _apply_skin_irradiance_cap(
    scene: SceneBuffers,
    source_d: np.ndarray | None,
    target_d: np.ndarray,
    target_s: np.ndarray,
    settings: EngineSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 5 (Irradiance cap): softly cap target/source ratio on skin_w pixels; strict no-op where skin_w == 0."""
    if source_d is None:
        return target_d, target_s
    skin_w = get_skin_weight_for_scene(scene, settings)
    if skin_w is None:
        return target_d, target_s
    sf = settings.skin_fidelity
    td_y = np.maximum(luminance(np.maximum(target_d, 0.0)), 1e-4)
    sd_y = np.maximum(luminance(np.maximum(source_d, 1e-4)), 1e-4)
    r_pix = td_y / sd_y
    s_start = float(sf.shoulder_start)
    r_ceil = float(sf.ratio_ceiling)
    span = max(r_ceil - s_start, 0.05)
    r_cap = np.where(r_pix > s_start, s_start + span * np.tanh((r_pix - s_start) / span), r_pix)
    skin_scale = np.where(r_pix > s_start, r_cap / r_pix, 1.0).astype(np.float32)
    eff_scale = ((1.0 - skin_w) * 1.0 + skin_w * skin_scale).astype(np.float32)
    spec_scale = ((1.0 - skin_w) + skin_w * (0.45 + 0.55 * skin_scale)).astype(np.float32)
    return (target_d * eff_scale[..., None]).astype(np.float32), (target_s * spec_scale[..., None]).astype(np.float32)


def _prepare_working_scene(
    scene: SceneBuffers,
    settings: EngineSettings,
    camera: CameraModel | None = None,
) -> SceneBuffers:
    """Apply non-destructive working-buffer enrichments (skirt clean, edge depth, normal-depth fusion, semantic PBR priors, albedo-guided shading)."""
    need_copy = bool(
        settings.refine_depth_edges
        or getattr(settings, "clean_depth_skirt", False)
        or (getattr(settings, "normal_depth_fusion", False) and scene.normals is not None and camera is not None)
        or (getattr(settings, "semantic_material_priors", False) and scene.roughness is None and scene.metalness is None)
        or getattr(settings, "auto_conductor_calibration", False)
        or (getattr(settings, "albedo_guided_source_shading", False) and scene.albedo is not None)
    )
    if not need_copy:
        return scene

    work_scene = copy(scene)
    if getattr(settings, "clean_depth_skirt", False):
        work_scene.depth = clean_depth_background_skirt(
            work_scene.depth, work_scene.rgb_srgb, work_scene.semantic_masks
        )
    if settings.refine_depth_edges:
        work_scene.depth = refine_depth_to_rgb_edges(work_scene.depth, work_scene.rgb_srgb)
    if getattr(settings, "normal_depth_fusion", False) and work_scene.normals is not None and camera is not None:
        work_scene.depth = integrate_normals_with_depth_prior(
            work_scene.depth, work_scene.normals, camera
        )

    if getattr(settings, "semantic_material_priors", False) and work_scene.roughness is None and work_scene.metalness is None:
        mat_maps = suggest_material_priors_from_semantics(work_scene)
        if mat_maps is not None:
            work_scene.roughness = mat_maps.roughness
            work_scene.metalness = mat_maps.metalness
            work_scene.specular_level = mat_maps.specular_level
            work_scene.clearcoat = mat_maps.clearcoat
            work_scene.clearcoat_roughness = mat_maps.clearcoat_roughness
            work_scene.subsurface = mat_maps.subsurface
            work_scene.anisotropy = mat_maps.anisotropy
            work_scene.anisotropy_angle = mat_maps.anisotropy_angle
            work_scene.material_confidence = mat_maps.confidence
            work_scene.material_metadata = dict(mat_maps.metadata)

    if getattr(settings, "auto_conductor_calibration", False):
        cond_res = calibrate_conductor_from_appearance(work_scene)
        if cond_res is not None:
            work_scene.metalness, work_scene.roughness = cond_res

    if getattr(settings, "albedo_guided_source_shading", False) and work_scene.albedo is not None:
        work_scene.source_shading = refine_source_shading_with_albedo(work_scene, work_scene.source_shading)

    return work_scene


def relight(
    scene: SceneBuffers,
    camera: CameraModel,
    source_rig: LightingRig,
    target_rig: LightingRig,
    settings: EngineSettings | None = None,
    *,
    require_geometry_ready: bool = False,
    neural_relighter=None,
) -> RelightResult:
    """Relight an image using shared 2.5D geometry and source-to-target transfer.

    Requires externally estimated depth and scene masks. Does not fabricate geometry.

    Set require_geometry_ready=True to enforce the structural normal-dependent
    lighting gate before rendering. The gate checks buffer coverage and reliability
    diagnostics; it does not certify that estimated geometry is physically correct.
    """
    settings = settings or EngineSettings()
    settings.validate()
    scene.validate()
    camera.validate()

    if settings.quality_mode == "preview":
        eff_settings = replace(
            settings,
            shadow_steps=min(settings.shadow_steps, 12),
            shadow_adaptive_sampling=False,
            shadow_smooth_passes=1,
        )
        preview_side = settings.preview_max_side if settings.preview_max_side is not None else 384
    elif settings.quality_mode == "final":
        eff_settings = replace(
            settings,
            shadow_steps=max(settings.shadow_steps, 48),
            shadow_adaptive_sampling=True,
            shadow_smooth_passes=max(settings.shadow_smooth_passes, 2),
        )
        preview_side = settings.preview_max_side
    else:
        eff_settings = settings
        preview_side = settings.preview_max_side

    work_scene = _prepare_working_scene(scene, eff_settings, camera=camera)

    use_preview_downscale = preview_side is not None and max(work_scene.shape) > int(preview_side)

    if use_preview_downscale and not require_geometry_ready:
        # Defer full-resolution geometry prep when rendering a downscaled non-upsampled preview directly.
        points, normals, geometry_conf = (None, None, None)
    else:
        points, normals, geometry_conf = prepare_geometry(work_scene, camera)
        eff_n_sigma = float(eff_settings.normal_smooth_sigma)
        if eff_n_sigma <= 0.0:
            valid_g = geometry_conf[work_scene.depth > 0]
            if valid_g.size > 0 and float(np.mean(valid_g)) < 0.55:
                eff_n_sigma = 2.0
        if eff_n_sigma > 0.0:
            smoothed = _depth_guided_smooth(normals, work_scene.depth, eff_n_sigma)
            lens = np.linalg.norm(smoothed, axis=-1, keepdims=True)
            valid_n = lens[..., 0] > 1e-6
            normals = np.where(valid_n[..., None], smoothed / np.maximum(lens, 1e-6), normals).astype(np.float32)
        if float(getattr(eff_settings, "micro_normal_bump", 0.0)) > 0.0:
            normals = extract_micro_normal_bump(
                normals,
                work_scene.rgb_srgb,
                strength=float(eff_settings.micro_normal_bump),
                roughness=work_scene.roughness,
            )

    gate_report = None
    if require_geometry_ready:
        assert normals is not None and geometry_conf is not None
        # QA the exact geometry prepared for this render, including normals estimated
        # on demand when they were not persisted in SceneBuffers.
        quality_scene = copy(scene)
        quality_scene.normals = normals
        quality_scene.geometry_confidence = geometry_conf
        gate_report = evaluate_scene_buffers(quality_scene)
        gate = gate_report["lighting_readiness"]["normal_dependent_shading"]
        if not gate["structural_gate_passed"]:
            reasons = "; ".join(gate["blocking_reasons"])
            raise ValueError(
                "Scene failed the structural gate for normal-dependent lighting: "
                + reasons
            )

    requested_mode = eff_settings.renderer_mode
    if requested_mode == "auto":
        effective_mode = "intrinsic" if scene.albedo is not None else "ratio"
    else:
        effective_mode = requested_mode

    if effective_mode == "intrinsic" and scene.albedo is None:
        raise ValueError("renderer_mode='intrinsic' requires scene.albedo to be populated.")
    if effective_mode == "hybrid_neural" and neural_relighter is None:
        raise ValueError("renderer_mode='hybrid_neural' requires a NeuralRelighter instance.")

    if use_preview_downscale:
        assert preview_side is not None
        eval_scene, eval_cam = _get_or_create_preview_scene(work_scene, camera, int(preview_side))
        eval_points, eval_normals, eval_geom_conf = prepare_geometry(eval_scene, eval_cam)
        if float(eff_settings.normal_smooth_sigma) > 0.0:
            smoothed_ev = _depth_guided_smooth(eval_normals, eval_scene.depth, float(eff_settings.normal_smooth_sigma))
            lens_ev = np.linalg.norm(smoothed_ev, axis=-1, keepdims=True)
            valid_ev = lens_ev[..., 0] > 1e-6
            eval_normals = np.where(valid_ev[..., None], smoothed_ev / np.maximum(lens_ev, 1e-6), eval_normals).astype(np.float32)
        if float(getattr(eff_settings, "micro_normal_bump", 0.0)) > 0.0:
            eval_normals = extract_micro_normal_bump(
                eval_normals,
                eval_scene.rgb_srgb,
                strength=float(eff_settings.micro_normal_bump),
                roughness=eval_scene.roughness,
            )
    else:
        eval_scene, eval_cam = work_scene, camera
        if points is None or normals is None or geometry_conf is None:
            points, normals, geometry_conf = prepare_geometry(work_scene, camera)
        eval_points, eval_normals, eval_geom_conf = points, normals, geometry_conf

    # Skin-fidelity shading pipeline order (strict no-op where skin_w == 0):
    # 1. Source floor (_effective_source_diffuse): removes skin albedo bias from eff_sd.
    # 2. Wrap (evaluate_rig): kept standard normalized wrap (0.0% error share in Phase 1 ablation).
    # 3. Bounce (compute_indirect_bounce * (1 - bounce_k * skin_w)): damps excess radiosity bounce on skin.
    # 4. Fill (preserve_fill_balance): kept standard (-9.3% negative error share in Phase 1 ablation).
    # 5. Irradiance cap (_apply_skin_irradiance_cap): compresses target/source ratio above shoulder_start on skin.
    add_mode = bool(eff_settings.preserve_source_lighting or eff_settings.operation == "add")
    skin_w_eval = get_skin_weight_for_scene(eval_scene, eff_settings)
    target_d, target_s, visibility, tgt_details = evaluate_rig(
        eval_scene, eval_cam, eval_points, eval_normals, target_rig, eff_settings,
        return_details=True,
    )
    weighted_vis = tgt_details["shadow_visibility_weighted"]
    target_d_raw = target_d
    target_s_raw = target_s
    if float(eff_settings.indirect_bounce_strength) > 0.0:
        bounce_d = compute_indirect_bounce(
            eval_scene, eval_points, eval_normals, target_d, eff_settings
        )
        target_d_raw = target_d + bounce_d
        if skin_w_eval is not None and float(eff_settings.skin_fidelity.bounce_k) > 0.0:
            bounce_d_skin = (bounce_d * (1.0 - float(eff_settings.skin_fidelity.bounce_k) * skin_w_eval[..., None])).astype(np.float32)
            target_d = target_d + bounce_d_skin
        else:
            target_d = target_d_raw
        if float(getattr(eff_settings, "indirect_specular_strength", 0.0)) > 0.0:
            spec_b = compute_indirect_specular(
                eval_scene, eval_points, eval_normals, bounce_d, eff_settings
            )
            target_s = target_s + spec_b
            target_s_raw = target_s_raw + spec_b
    if float(getattr(eff_settings, "volumetric_beam_strength", 0.0)) > 0.0:
        v_beam = compute_volumetric_beam(
            eval_scene, eval_points, target_rig, visibility, eff_settings
        )
        target_d = target_d + v_beam
        target_d_raw = target_d_raw + v_beam
    source_d: np.ndarray | None = None
    source_s: np.ndarray | None = None
    source_d_off: np.ndarray | None = None
    need_source_eval = (
        add_mode
        or effective_mode in ("ratio", "intrinsic")
        or float(eff_settings.background_light_influence) < 1.0 - 1e-6
        or float(eff_settings.real_image_preservation) > 0.0
        or float(eff_settings.highlight_relocation_strength) > 0.0
    )
    src_exp = float(eff_settings.source_exposure_scale)
    src_wb = np.asarray(eff_settings.source_white_balance, dtype=np.float32)[None, None, :]
    if need_source_eval:
        if (not add_mode) and _rigs_identical(source_rig, target_rig) and abs(src_exp - 1.0) < 1e-6 and np.allclose(src_wb, 1.0, atol=1e-6):
            source_d = target_d
            source_s = target_s
            source_d_off = target_d_raw
        else:
            source_d_raw, source_s, _ = evaluate_rig(
                eval_scene, eval_cam, eval_points, eval_normals, source_rig, eff_settings
            )
            source_d_raw = (source_d_raw * src_exp * src_wb).astype(np.float32)
            source_s = (source_s * src_exp * src_wb).astype(np.float32)
            if skin_w_eval is not None:
                eff_off_cfg = replace(eff_settings, skin_fidelity=replace(eff_settings.skin_fidelity, enabled=False))
                source_d_off = _effective_source_diffuse(eval_scene, eval_normals, source_d_raw, eff_off_cfg)
            source_d = _effective_source_diffuse(eval_scene, eval_normals, source_d_raw, eff_settings)
    if add_mode:
        assert source_d is not None and source_s is not None
        target_d = source_d + target_d
        target_s = source_s + target_s
    elif (not _rigs_identical(source_rig, target_rig)) and source_d is not None:
        target_d, target_s = _apply_skin_irradiance_cap(eval_scene, source_d, target_d, target_s, eff_settings)

    if use_preview_downscale and eff_settings.preview_upsample:
        if points is None or normals is None or geometry_conf is None:
            points, normals, geometry_conf = prepare_geometry(work_scene, camera)
            if float(eff_settings.normal_smooth_sigma) > 0.0:
                smoothed = _depth_guided_smooth(normals, work_scene.depth, float(eff_settings.normal_smooth_sigma))
                lens = np.linalg.norm(smoothed, axis=-1, keepdims=True)
                valid_n = lens[..., 0] > 1e-6
                normals = np.where(valid_n[..., None], smoothed / np.maximum(lens, 1e-6), normals).astype(np.float32)
        to_up = [target_d, target_s, visibility, weighted_vis]
        if source_d is not None:
            to_up.append(source_d)
        if source_s is not None:
            to_up.append(source_s)
        up = depth_guided_upsample_batch(to_up, work_scene.depth, eval_scene.depth)
        target_d, target_s, visibility, weighted_vis = up[0], up[1], up[2], up[3]
        idx = 4
        if source_d is not None:
            source_d = up[idx]
            idx += 1
        if source_s is not None:
            source_s = up[idx]
        source_d_off_up = source_d_off
        if source_d_off is not None and (not add_mode):
            up_off = depth_guided_upsample_batch([target_d_raw, target_s_raw, source_d_off], work_scene.depth, eval_scene.depth)
            target_d_raw, target_s_raw, source_d_off_up = up_off[0], up_off[1], up_off[2]
        comp_scene = work_scene
        comp_normals = normals
        comp_geom_conf = geometry_conf
        metric_scene = scene
    elif use_preview_downscale:
        source_d_off_up = source_d_off
        comp_scene = eval_scene
        comp_normals = eval_normals
        comp_geom_conf = eval_geom_conf
        metric_scene = eval_scene
    else:
        source_d_off_up = source_d_off
        comp_scene = work_scene
        assert normals is not None and geometry_conf is not None
        comp_normals = normals
        comp_geom_conf = geometry_conf
        metric_scene = scene

    neural_metadata = None
    if effective_mode == "intrinsic":
        out, ratio = compose_intrinsic(
            comp_scene,
            target_d,
            target_s,
            comp_geom_conf,
            eff_settings,
            source_diffuse=source_d,
            source_specular=source_s,
            target_rig=target_rig,
        )
        if skin_w_eval is not None and (not add_mode) and source_d_off_up is not None:
            eff_off_cfg = replace(eff_settings, skin_fidelity=replace(eff_settings.skin_fidelity, enabled=False))
            out_off, _ = compose_intrinsic(
                comp_scene, target_d_raw, target_s_raw, comp_geom_conf, eff_off_cfg,
                source_diffuse=source_d_off_up, source_specular=source_s, target_rig=target_rig,
            )
            sw_c = get_skin_weight_for_scene(comp_scene, eff_settings)
            if sw_c is not None:
                out = np.where(sw_c[..., None] > 0.0, (1.0 - sw_c[..., None]) * out_off + sw_c[..., None] * out, out_off).astype(np.float32)
        mode_label = "2.5D intrinsic de-lighting and re-rendering"
    elif effective_mode == "hybrid_neural":
        neural_pred = neural_relighter.render(
            comp_scene, target_d, target_s, target_rig, eff_settings
        )
        neural_rgb = np.asarray(neural_pred.rgb_srgb, dtype=np.float32)
        if neural_rgb.shape != comp_scene.rgb_srgb.shape:
            raise ValueError(
                f"NeuralRelighter returned shape {neural_rgb.shape}; expected {comp_scene.rgb_srgb.shape}."
            )
        if not np.isfinite(neural_rgb).all() or np.any(neural_rgb < 0.0) or np.any(neural_rgb > 1.0):
            raise ValueError("NeuralRelighter returned non-finite or out-of-[0,1] pixel values.")
        out = compose_hybrid_neural(comp_scene, neural_rgb, comp_geom_conf, eff_settings)
        t_y = np.maximum(luminance(target_d), 1e-4)
        if comp_scene.source_shading is not None and not eff_settings.preserve_source_lighting:
            sh = comp_scene.source_shading[..., None] if comp_scene.source_shading.ndim == 2 else comp_scene.source_shading
            s_y = np.maximum(luminance(np.broadcast_to(sh, (*comp_scene.shape, 3))), 1e-4)
        else:
            if source_d is None:
                source_d, _, _ = evaluate_rig(
                    comp_scene, camera if not (use_preview_downscale and not eff_settings.preview_upsample) else eval_cam,
                    points if comp_scene is work_scene else eval_points, comp_normals, source_rig, eff_settings
                )
            s_y = np.maximum(luminance(source_d), 1e-4)
        ratio = np.clip(t_y / s_y, eff_settings.ratio_min, eff_settings.ratio_max).astype(np.float32)
        neural_metadata = dict(getattr(neural_pred, "metadata", {}) or {})
        mode_label = "Hybrid physics-steered neural relighting with detail lock"
    else:
        assert source_d is not None and source_s is not None
        out, ratio = compose_source_preserving(
            comp_scene,
            source_d,
            target_d,
            source_s,
            target_s,
            comp_geom_conf,
            eff_settings,
            target_rig=target_rig,
        )
        if skin_w_eval is not None and (not add_mode) and source_d_off_up is not None:
            eff_off_cfg = replace(eff_settings, skin_fidelity=replace(eff_settings.skin_fidelity, enabled=False))
            out_off, _ = compose_source_preserving(
                comp_scene, source_d_off_up, target_d_raw, source_s, target_s_raw,
                comp_geom_conf, eff_off_cfg, target_rig=target_rig,
            )
            sw_c = get_skin_weight_for_scene(comp_scene, eff_settings)
            if sw_c is not None:
                out = np.where(sw_c[..., None] > 0.0, (1.0 - sw_c[..., None]) * out_off + sw_c[..., None] * out, out_off).astype(np.float32)
        mode_label = "2.5D source-to-target relighting"

    active = (
        (metric_scene.lightable_mask > 0.05)
        & (comp_geom_conf > eff_settings.min_geometry_confidence)
    )
    if metric_scene.protected_mask is not None:
        active = active & (metric_scene.protected_mask < 0.95)
    if metric_scene.alpha_mask is not None:
        active = active & (metric_scene.alpha_mask > 0.05)
    delta = np.abs(out - metric_scene.rgb_srgb)
    valid_d = metric_scene.depth > 0
    depth_span = (
        [float(np.min(metric_scene.depth[valid_d])), float(np.max(metric_scene.depth[valid_d]))]
        if np.any(valid_d)
        else [0.0, 0.0]
    )
    metrics = {
        "mode": mode_label,
        "renderer_mode": effective_mode,
        **_pipeline_observability(scene, requested_mode, effective_mode, target_rig),
        "quality_mode": eff_settings.quality_mode,
        "shadow_steps_effective": int(eff_settings.shadow_steps),
        "preview_downscaled": bool(use_preview_downscale),
        "preview_upsampled": bool(use_preview_downscale and eff_settings.preview_upsample),
        "depth_kind": scene.depth_kind,
        "lighting_scale": {
            "depth_kind": scene.depth_kind,
            "calibrated_metric_scale": scene.depth_kind == "metric",
            "attenuation_law": (
                "inverse_square_metres (strength / max(d^2, 0.04))"
                if scene.depth_kind == "metric"
                else "bounded_relative_dimensionless (strength / (1 + 0.15 * d^2))"
            ),
            "depth_units": "metres" if scene.depth_kind == "metric" else "relative_scene_units",
            "depth_span": depth_span,
            "base_shadow_bias": (
                eff_settings.shadow_bias_metric
                if scene.depth_kind == "metric"
                else eff_settings.shadow_bias_relative
            ),
        },
        "material_response": describe_material_response(scene),
        "neural_relighter": neural_metadata,
        "geometry_valid_fraction": float(
            np.mean(
                (metric_scene.depth > 0)
                & (comp_geom_conf > eff_settings.min_geometry_confidence)
            )
        ),
        "lightable_fraction": float(np.mean(active)),
        "mean_abs_srgb_change_active": (
            float(np.mean(delta[active])) if np.any(active) else 0.0
        ),
        "mean_abs_srgb_change_protected": (
            float(np.mean(delta[~active])) if np.any(~active) else 0.0
        ),
        "mean_target_to_source_shading_ratio": (
            float(np.mean(ratio[active])) if np.any(active) else 1.0
        ),
        "shadow_visibility_mean": (
            float(np.mean(visibility[active])) if np.any(active) else 1.0
        ),
        "shadow_visibility_weighted_mean": (
            float(np.mean(weighted_vis[active])) if np.any(active) else 1.0
        ),
        "operation": "add" if add_mode else "replace",
        "per_light_contributions": tgt_details["per_light_contributions"],
        "per_light_visibility_mean": {
            lid: (float(np.mean(v[active])) if np.any(active) else 1.0)
            for lid, v in tgt_details["per_light_visibility"].items()
        } if not (use_preview_downscale and eff_settings.preview_upsample) else {
            rec["light_id"]: rec["visibility_mean"] for rec in tgt_details["per_light_contributions"]
        },
        "geometry_readiness_gate_enforced": bool(require_geometry_ready),
        "geometry_readiness_gate": (
            gate_report["lighting_readiness"]["normal_dependent_shading"]
            if gate_report is not None
            else None
        ),
        "neural_scene_estimation_used": bool(
            scene.metadata.get("neural_scene_estimation_used", False)
        ),
        "preservation": evaluate_preservation(comp_scene, out, target_alpha=comp_scene.alpha_mask),
        "limitations": [
            "2.5D depth represents only the visible first surface; hidden surfaces and overhangs are unavailable.",
            "Relighting quality depends on depth, camera calibration, semantic masks, and source-light estimation.",
            "Shadow visibility is approximate screen-space ray marching, not a full 3D mesh shadow map.",
            "Single-image intrinsic decomposition remains ambiguous; source_rig must approximate the original illumination.",
            "The structural geometry gate is not a substitute for real-photo visual validation.",
            "Relative depth does not support metre-based point/area-light distance falloff.",
            "Direct lighting and parameterized studio environment probes do not model mirror raytraced inter-reflections, glass/water refraction, or transmission.",
        ],
    }
    return RelightResult(
        out, comp_normals, ratio, visibility, comp_geom_conf, metrics
    )


class RelightSession:
    """Reusable scene session avoiding redundant buffer validation, hashing, and geometry prep.

    Designed for interactive light manipulation: construct once when a scene loads,
    then call `render(target_rig, ...)` repeatedly as lights move or change colour.
    """

    def __init__(
        self,
        scene: SceneBuffers,
        camera: CameraModel,
        source_rig: LightingRig,
        settings: EngineSettings | None = None,
        *,
        preview_max_side: int = 384,
        require_geometry_ready: bool = False,
    ) -> None:
        base_settings = settings or EngineSettings()
        base_settings.validate()
        scene.validate()
        camera.validate()
        source_rig.validate()
        if base_settings.renderer_mode == "hybrid_neural":
            raise ValueError(
                "RelightSession does not support renderer_mode='hybrid_neural'; use relight(..., neural_relighter=...) outside the deterministic preservation session."
            )

        self.scene = scene
        self.camera = camera
        self.source_rig = source_rig
        self.base_settings = base_settings
        self.preview_max_side = max(32, int(preview_max_side))

        work_scene = _prepare_working_scene(scene, base_settings, camera=camera)
        self.work_scene = work_scene

        points, normals, geom_conf = prepare_geometry(work_scene, camera)
        eff_n_sigma = float(base_settings.normal_smooth_sigma)
        if eff_n_sigma <= 0.0:
            valid_g = geom_conf[work_scene.depth > 0]
            if valid_g.size > 0 and float(np.mean(valid_g)) < 0.55:
                eff_n_sigma = 2.0
        if eff_n_sigma > 0.0:
            smoothed = _depth_guided_smooth(normals, work_scene.depth, eff_n_sigma)
            lens = np.linalg.norm(smoothed, axis=-1, keepdims=True)
            valid_n = lens[..., 0] > 1e-6
            normals = np.where(valid_n[..., None], smoothed / np.maximum(lens, 1e-6), normals).astype(np.float32)
        if float(getattr(base_settings, "micro_normal_bump", 0.0)) > 0.0:
            normals = extract_micro_normal_bump(
                normals,
                work_scene.rgb_srgb,
                strength=float(base_settings.micro_normal_bump),
                roughness=work_scene.roughness,
            )
        self.points = points
        self.normals = normals
        self.geom_conf = geom_conf

        if require_geometry_ready:
            quality_scene = copy(scene)
            quality_scene.normals = normals
            quality_scene.geometry_confidence = geom_conf
            gate_report = evaluate_scene_buffers(quality_scene)
            gate = gate_report["lighting_readiness"]["normal_dependent_shading"]
            if not gate["structural_gate_passed"]:
                raise ValueError(
                    "Scene failed the structural gate for normal-dependent lighting: "
                    + "; ".join(gate["blocking_reasons"])
                )

        # Precompute full-res compositing invariants (F3 priors, multiscale residual layers)
        self.material_meta = describe_material_response(scene)
        self.albedo_corr_hi: np.ndarray | None = None
        self.detail_factor_hi: np.ndarray | None = None
        self.residual_layers_hi: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if work_scene.albedo is not None:
            self.albedo_corr_hi, self.detail_factor_hi = _get_or_compute_intrinsic_priors(work_scene, base_settings)
            self.residual_layers_hi = _get_or_compute_intrinsic_residual(work_scene, base_settings)

        # Precompute preview buffers & geometry for fast scrubbing
        self.prev_scene, self.prev_cam = downsample_scene_buffers_for_preview(work_scene, camera, self.preview_max_side)
        self.prev_points, prev_norms, self.prev_geom_conf = prepare_geometry(self.prev_scene, self.prev_cam)
        if float(base_settings.normal_smooth_sigma) > 0.0:
            smoothed_p = _depth_guided_smooth(prev_norms, self.prev_scene.depth, float(base_settings.normal_smooth_sigma))
            lens_p = np.linalg.norm(smoothed_p, axis=-1, keepdims=True)
            valid_p = lens_p[..., 0] > 1e-6
            prev_norms = np.where(valid_p[..., None], smoothed_p / np.maximum(lens_p, 1e-6), prev_norms).astype(np.float32)
        self.prev_normals = prev_norms
        self.albedo_corr_lo: np.ndarray | None = None
        self.detail_factor_lo: np.ndarray | None = None
        self.residual_layers_lo: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if self.prev_scene.albedo is not None:
            self.albedo_corr_lo, self.detail_factor_lo = _get_or_compute_intrinsic_priors(self.prev_scene, base_settings)
            self.residual_layers_lo = _get_or_compute_intrinsic_residual(self.prev_scene, base_settings)

    def render(
        self,
        target_rig: LightingRig,
        settings: EngineSettings | None = None,
    ) -> RelightResult:
        """Render using cached geometry and intrinsic priors for responsive interaction."""
        cfg = settings or self.base_settings
        cfg.validate()
        target_rig.validate()

        requested_mode = cfg.renderer_mode
        if requested_mode == "hybrid_neural":
            raise ValueError(
                "RelightSession does not support renderer_mode='hybrid_neural'; use relight(..., neural_relighter=...) outside the deterministic preservation session."
            )
        effective_mode = ("intrinsic" if self.scene.albedo is not None else "ratio") if requested_mode == "auto" else requested_mode
        if effective_mode == "intrinsic" and self.scene.albedo is None:
            raise ValueError("renderer_mode='intrinsic' requires scene.albedo to be populated.")

        if cfg.quality_mode == "preview":
            eff = replace(cfg, shadow_steps=min(cfg.shadow_steps, 12), shadow_adaptive_sampling=False, shadow_smooth_passes=1)
            use_prev = max(self.work_scene.shape) > (cfg.preview_max_side or self.preview_max_side)
        elif cfg.quality_mode == "final":
            eff = replace(cfg, shadow_steps=max(cfg.shadow_steps, 48), shadow_adaptive_sampling=True, shadow_smooth_passes=max(cfg.shadow_smooth_passes, 2))
            use_prev = False
        else:
            eff = cfg
            use_prev = cfg.preview_max_side is not None and max(self.work_scene.shape) > int(cfg.preview_max_side)

        if use_prev:
            ev_sc, ev_cam, ev_pts, ev_nrm = self.prev_scene, self.prev_cam, self.prev_points, self.prev_normals
        else:
            ev_sc, ev_cam, ev_pts, ev_nrm = self.work_scene, self.camera, self.points, self.normals

        add_mode = bool(eff.preserve_source_lighting or eff.operation == "add")
        skin_w_ev = get_skin_weight_for_scene(ev_sc, eff)
        target_d, target_s, visibility, tgt_details = evaluate_rig(
            ev_sc, ev_cam, ev_pts, ev_nrm, target_rig, eff, return_details=True
        )
        weighted_vis = tgt_details["shadow_visibility_weighted"]
        if float(eff.indirect_bounce_strength) > 0.0:
            bounce_d = compute_indirect_bounce(ev_sc, ev_pts, ev_nrm, target_d, eff)
            if skin_w_ev is not None and float(eff.skin_fidelity.bounce_k) > 0.0:
                bounce_d = (bounce_d * (1.0 - float(eff.skin_fidelity.bounce_k) * skin_w_ev[..., None])).astype(np.float32)
            target_d = target_d + bounce_d
            if float(getattr(eff, "indirect_specular_strength", 0.0)) > 0.0:
                target_s = target_s + compute_indirect_specular(ev_sc, ev_pts, ev_nrm, bounce_d, eff)
        if float(getattr(eff, "volumetric_beam_strength", 0.0)) > 0.0:
            target_d = target_d + compute_volumetric_beam(ev_sc, ev_pts, target_rig, visibility, eff)

        source_d: np.ndarray | None = None
        source_s: np.ndarray | None = None
        need_source_eval = (
            add_mode
            or effective_mode in ("ratio", "intrinsic")
            or float(eff.background_light_influence) < 1.0 - 1e-6
            or float(eff.real_image_preservation) > 0.0
            or float(eff.highlight_relocation_strength) > 0.0
        )
        src_exp = float(eff.source_exposure_scale)
        src_wb = np.asarray(eff.source_white_balance, dtype=np.float32)[None, None, :]
        if need_source_eval:
            if (not add_mode) and _rigs_identical(self.source_rig, target_rig) and abs(src_exp - 1.0) < 1e-6 and np.allclose(src_wb, 1.0, atol=1e-6):
                source_d = target_d
                source_s = target_s
            else:
                source_d, source_s, _ = evaluate_rig(ev_sc, ev_cam, ev_pts, ev_nrm, self.source_rig, eff)
                source_d = (source_d * src_exp * src_wb).astype(np.float32)
                source_s = (source_s * src_exp * src_wb).astype(np.float32)
                source_d = _effective_source_diffuse(ev_sc, ev_nrm, source_d, eff)
        if add_mode:
            assert source_d is not None and source_s is not None
            target_d = source_d + target_d
            target_s = source_s + target_s
        elif (not _rigs_identical(self.source_rig, target_rig)) and source_d is not None:
            target_d, target_s = _apply_skin_irradiance_cap(ev_sc, source_d, target_d, target_s, eff)

        if use_prev and eff.preview_upsample:
            to_up = [target_d, target_s, visibility, weighted_vis]
            if source_d is not None:
                to_up.append(source_d)
            if source_s is not None:
                to_up.append(source_s)
            up = depth_guided_upsample_batch(to_up, self.work_scene.depth, self.prev_scene.depth)
            target_d, target_s, visibility, weighted_vis = up[0], up[1], up[2], up[3]
            idx = 4
            if source_d is not None:
                source_d = up[idx]
                idx += 1
            if source_s is not None:
                source_s = up[idx]
            hi_res = True
        elif use_prev:
            hi_res = False
        else:
            hi_res = True

        comp_sc = self.work_scene if hi_res else self.prev_scene
        comp_nrm = self.normals if hi_res else self.prev_normals
        comp_conf = self.geom_conf if hi_res else self.prev_geom_conf
        eps = 1e-4

        if effective_mode == "intrinsic":
            priors_valid = (
                cfg.detail_preservation_mode == self.base_settings.detail_preservation_mode
                and cfg.detail_lock_strength == self.base_settings.detail_lock_strength
                and cfg.base_blur_sigma == self.base_settings.base_blur_sigma
                and cfg.edge_aware_base == self.base_settings.edge_aware_base
            )
            if hi_res and self.albedo_corr_hi is not None and self.detail_factor_hi is not None and priors_valid:
                alb_corr, det_fac, res_layers = (
                    self.albedo_corr_hi, self.detail_factor_hi, self.residual_layers_hi
                )
            elif (not hi_res) and self.albedo_corr_lo is not None and self.detail_factor_lo is not None and priors_valid:
                alb_corr, det_fac, res_layers = (
                    self.albedo_corr_lo, self.detail_factor_lo, self.residual_layers_lo
                )
            else:
                alb_corr, det_fac = _get_or_compute_intrinsic_priors(comp_sc, eff, eps=eps)
                res_layers = _get_or_compute_intrinsic_residual(comp_sc, eff, eps=eps)

            out, ratio = _compose_intrinsic_output(
                comp_sc,
                alb_corr,
                det_fac,
                res_layers,
                target_d,
                target_s,
                comp_conf,
                eff,
                eps=eps,
                source_diffuse=source_d,
                source_specular=source_s,
                target_rig=target_rig,
            )
        else:
            assert source_d is not None and source_s is not None
            out, ratio = compose_source_preserving(
                comp_sc, source_d, target_d, source_s, target_s, comp_conf, eff, target_rig=target_rig
            )

        metrics = {
            "mode": "2.5D intrinsic de-lighting and re-rendering" if effective_mode == "intrinsic" else "2.5D source-to-target relighting",
            "renderer_mode": effective_mode,
            **_pipeline_observability(self.scene, requested_mode, effective_mode, target_rig),
            "quality_mode": eff.quality_mode,
            "shadow_steps_effective": int(eff.shadow_steps),
            "preview_downscaled": bool(use_prev),
            "preview_upsampled": bool(use_prev and eff.preview_upsample),
            "depth_kind": self.scene.depth_kind,
            "material_response": self.material_meta,
            "shadow_visibility_mean": float(np.mean(visibility)),
            "shadow_visibility_weighted_mean": float(np.mean(weighted_vis)),
            "operation": "add" if add_mode else "replace",
            "per_light_contributions": tgt_details["per_light_contributions"],
            "preservation": evaluate_preservation(comp_sc, out, target_alpha=comp_sc.alpha_mask),
            "session_cached": True,
        }
        return RelightResult(out, comp_nrm, ratio, visibility, comp_conf, metrics)


def prepare_relight_session(
    scene: SceneBuffers,
    camera: CameraModel,
    source_rig: LightingRig,
    settings: EngineSettings | None = None,
    *,
    preview_max_side: int = 384,
    require_geometry_ready: bool = False,
) -> RelightSession:
    """Create a reusable RelightSession with cached geometry and intrinsic priors."""
    return RelightSession(
        scene,
        camera,
        source_rig,
        settings,
        preview_max_side=preview_max_side,
        require_geometry_ready=require_geometry_ready,
    )
