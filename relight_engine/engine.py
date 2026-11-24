from __future__ import annotations

from copy import copy

import numpy as np

from .contracts import (
    CameraModel,
    SceneBuffers,
    LightingRig,
    EngineSettings,
    RelightResult,
)
from .geometry import prepare_geometry
from .lighting import evaluate_rig
from .composite import compose_hybrid_neural, compose_intrinsic, compose_source_preserving
from .quality import evaluate_scene_buffers, describe_material_response
from .color import luminance


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

    points, normals, geometry_conf = prepare_geometry(scene, camera)

    gate_report = None
    if require_geometry_ready:
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

    requested_mode = settings.renderer_mode
    effective_mode = "ratio" if requested_mode == "auto" else requested_mode

    if effective_mode == "intrinsic" and scene.albedo is None:
        raise ValueError("renderer_mode='intrinsic' requires scene.albedo to be populated.")
    if effective_mode == "hybrid_neural" and neural_relighter is None:
        raise ValueError("renderer_mode='hybrid_neural' requires a NeuralRelighter instance.")

    target_d, target_s, visibility = evaluate_rig(
        scene, camera, points, normals, target_rig, settings
    )
    source_d: np.ndarray | None = None
    source_s: np.ndarray | None = None
    if settings.preserve_source_lighting or effective_mode == "ratio":
        source_d, source_s, _ = evaluate_rig(
            scene, camera, points, normals, source_rig, settings
        )
    if settings.preserve_source_lighting:
        assert source_d is not None and source_s is not None
        target_d = source_d + target_d
        target_s = source_s + target_s

    neural_metadata = None
    if effective_mode == "intrinsic":
        out, ratio = compose_intrinsic(
            scene, target_d, target_s, geometry_conf, settings
        )
        mode_label = "2.5D intrinsic de-lighting and re-rendering"
    elif effective_mode == "hybrid_neural":
        neural_pred = neural_relighter.render(
            scene, target_d, target_s, target_rig, settings
        )
        neural_rgb = np.asarray(neural_pred.rgb_srgb, dtype=np.float32)
        if neural_rgb.shape != scene.rgb_srgb.shape:
            raise ValueError(
                f"NeuralRelighter returned shape {neural_rgb.shape}; expected {scene.rgb_srgb.shape}."
            )
        if not np.isfinite(neural_rgb).all() or np.any(neural_rgb < 0.0) or np.any(neural_rgb > 1.0):
            raise ValueError("NeuralRelighter returned non-finite or out-of-[0,1] pixel values.")
        out = compose_hybrid_neural(scene, neural_rgb, geometry_conf, settings)
        t_y = np.maximum(luminance(target_d), 1e-4)
        if scene.source_shading is not None and not settings.preserve_source_lighting:
            sh = scene.source_shading[..., None] if scene.source_shading.ndim == 2 else scene.source_shading
            s_y = np.maximum(luminance(np.broadcast_to(sh, (*scene.shape, 3))), 1e-4)
        else:
            if source_d is None:
                source_d, _, _ = evaluate_rig(
                    scene, camera, points, normals, source_rig, settings
                )
            s_y = np.maximum(luminance(source_d), 1e-4)
        ratio = np.clip(t_y / s_y, settings.ratio_min, settings.ratio_max).astype(np.float32)
        neural_metadata = dict(getattr(neural_pred, "metadata", {}) or {})
        mode_label = "Hybrid physics-steered neural relighting with detail lock"
    else:
        assert source_d is not None and source_s is not None
        out, ratio = compose_source_preserving(
            scene,
            source_d,
            target_d,
            source_s,
            target_s,
            geometry_conf,
            settings,
        )
        mode_label = "2.5D source-to-target relighting"

    active = (
        (scene.lightable_mask > 0.05)
        & (geometry_conf > settings.min_geometry_confidence)
    )
    if scene.protected_mask is not None:
        active = active & (scene.protected_mask < 0.95)
    if scene.alpha_mask is not None:
        active = active & (scene.alpha_mask > 0.05)
    delta = np.abs(out - scene.rgb_srgb)
    metrics = {
        "mode": mode_label,
        "renderer_mode": effective_mode,
        "depth_kind": scene.depth_kind,
        "material_response": describe_material_response(scene),
        "neural_relighter": neural_metadata,
        "geometry_valid_fraction": float(
            np.mean(
                (scene.depth > 0)
                & (geometry_conf > settings.min_geometry_confidence)
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
        "geometry_readiness_gate_enforced": bool(require_geometry_ready),
        "geometry_readiness_gate": (
            gate_report["lighting_readiness"]["normal_dependent_shading"]
            if gate_report is not None
            else None
        ),
        "neural_scene_estimation_used": bool(
            scene.metadata.get("neural_scene_estimation_used", False)
        ),
        "limitations": [
            "2.5D depth represents only the visible first surface; hidden surfaces and overhangs are unavailable.",
            "Relighting quality depends on depth, camera calibration, semantic masks, and source-light estimation.",
            "Shadow visibility is approximate screen-space ray marching, not a full 3D mesh shadow map.",
            "Single-image intrinsic decomposition remains ambiguous; source_rig must approximate the original illumination.",
            "The structural geometry gate is not a substitute for real-photo visual validation.",
            "Relative depth does not support metre-based point/area-light distance falloff.",
        ],
    }
    return RelightResult(
        out, normals, ratio, visibility, geometry_conf, metrics
    )
