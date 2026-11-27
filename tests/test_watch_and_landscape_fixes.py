"""Regression tests for watch sharpness/dial priors, finger silhouette shadows, and landscape light scale."""
from __future__ import annotations

import numpy as np

from relight_engine import (
    CameraModel,
    EngineSettings,
    Light,
    LightingRig,
    SceneBuffers,
    classify_scene_scale,
    place_studio_light,
    suggest_material_priors_from_semantics,
)
from relight_engine.geometry import backproject_depth
from relight_engine.lighting import compute_volumetric_beam
from relight_engine.shadows import ray_visibility
from relight_engine.wavelet import compose_wavelet_identity_lock


def test_semantic_priors_keep_watch_dial_dielectric_and_avoid_border_metal_bleed():
    h, w = 48, 48
    watch = np.zeros((h, w), dtype=np.float32)
    watch[12:36, 12:36] = 1.0
    # Soft blurred border tail outside watch onto neighboring finger pixels
    watch[10:12, 12:36] = 0.28
    dial = np.zeros((h, w), dtype=np.float32)
    dial[18:30, 18:30] = 1.0
    markers = np.zeros((h, w), dtype=np.float32)
    markers[20:22, 20:28] = 1.0

    scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.60, dtype=np.float32),
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        semantic_masks={"watch": watch, "black_dial": dial, "markers": markers},
    )
    mats = suggest_material_priors_from_semantics(scene)
    assert mats is not None
    # Watch bezel stays metallic, dial stays dielectric (clearcoat glass/lacquer), soft halo rim is zeroed
    assert float(mats.metalness[14, 24]) > 0.70
    assert float(mats.metalness[24, 24]) < 0.12
    assert mats.clearcoat is not None and float(mats.clearcoat[24, 24]) > 0.50
    assert float(mats.metalness[11, 24]) < 0.05


def test_wavelet_identity_lock_preserves_high_freq_text_on_darkened_relit():
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // 2 + xx // 2) & 1) == 0).astype(np.float32)
    src = np.repeat((0.20 + 0.45 * checker)[..., None], 3, axis=-1).astype(np.float32)
    # Simulate darkened relit base where fine texture must not collapse
    relit = (src * 0.45).astype(np.float32)
    out = compose_wavelet_identity_lock(src, relit, levels=4, detail_lock_strength=0.92, mid_blend=0.70)

    def _lap_rms(arr: np.ndarray) -> float:
        g = arr[..., 0]
        lap = g[1:-1, 1:-1] * 4.0 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        return float(np.sqrt(np.mean(lap * lap)))

    assert _lap_rms(out) > 0.55 * _lap_rms(src)


def test_curved_foreground_limb_does_not_carve_black_silhouette_outline():
    h, w = 64, 64
    cam = CameraModel(64.0, 64.0, w / 2.0, h / 2.0)
    depth = np.full((h, w), 2.8, dtype=np.float32)
    # Curved finger cylinder in foreground Z ~ 1.75..1.90 overlooking background at 2.8
    yy, xx = np.mgrid[0:h, 0:w]
    dx = (xx - 32.0) / 10.0
    in_finger = (np.abs(dx) <= 1.0) & (yy >= 12) & (yy <= 52)
    nz_cyl = np.where(in_finger, np.sqrt(np.maximum(1.0 - dx * dx, 0.05)), 1.0)
    depth = np.where(in_finger, 1.85 - 0.14 * nz_cyl, depth).astype(np.float32)

    points = backproject_depth(depth, cam)
    normals = np.zeros_like(points)
    normals[..., 2] = -1.0
    normals[in_finger, 0] = dx[in_finger] * 0.85
    normals[in_finger, 2] = -np.sqrt(np.maximum(1.0 - normals[in_finger, 0] ** 2, 0.05))

    to_light = np.zeros_like(points)
    to_light[:] = np.array([-0.8, 0.25, -0.55], dtype=np.float32)
    vis = ray_visibility(
        points, normals, depth, cam, to_light, np.full((h, w), 3.0, np.float32),
        "relative", EngineSettings.studio_3d(shadow_max_thickness=1.2), light_radius=0.25,
    )
    # Rim pixels on the curved finger must not collapse to near-black self-occlusion
    rim = in_finger & (np.abs(dx) >= 0.85)
    assert float(vis[rim].min()) >= 0.60


def test_landscape_scale_detection_places_light_above_ridges_and_suppresses_valley_beam():
    h, w = 64, 64
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    depth = np.full((h, w), 4.8, dtype=np.float32)
    depth[20:42, :] = np.linspace(4.2, 2.4, 22, dtype=np.float32)[:, None]
    depth[42:, :] = np.linspace(2.4, 1.3, 22, dtype=np.float32)[:, None]

    sky = np.zeros((h, w), dtype=np.float32)
    sky[:20, :] = 1.0
    mtn = np.zeros((h, w), dtype=np.float32)
    mtn[20:, :] = 1.0
    hiker = np.zeros((h, w), dtype=np.float32)
    hiker[48:55, 40:44] = 1.0  # tiny ~1% foreground subject

    scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.5, dtype=np.float32),
        depth=depth,
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        semantic_masks={"clear_blue_sky": sky, "majestic_mountains": mtn, "person": hiker},
    )
    info = classify_scene_scale(scene)
    assert info["is_landscape"] is True

    pts = backproject_depth(depth, cam)
    pos, _ = place_studio_light(scene, cam, azimuth_deg=-48.0, elevation_deg=-20.0, distance_scale=1.15)
    ridge_top_y = float(np.percentile(pts[mtn > 0.5, 1], 95))
    # Sky/sun light must sit cleanly above the mountain ridge horizon rather than inside the valley floor
    assert pos[1] > ridge_top_y

    rig = LightingRig(
        lights=(Light("valley_spot", "point", position=(0.0, -0.8, 1.6), color=(1.0, 0.7, 0.4), strength=3.0, casts_shadows=False),)
    )
    beam = compute_volumetric_beam(scene, pts, rig, np.ones((h, w), np.float32), EngineSettings.studio_3d())
    assert float(beam.max()) == 0.0


def test_convex_bilateral_smooth_has_zero_overshoot_across_depth_cliffs():
    from relight_engine.composite import _depth_guided_smooth

    h, w = 48, 48
    depth = np.full((h, w), 4.0, dtype=np.float32)
    depth[:, :24] = 1.8
    field = np.full((h, w), 0.35, dtype=np.float32)
    field[:, :24] = 1.25
    smoothed = _depth_guided_smooth(field, depth, 2.0)
    assert float(smoothed.min()) >= 0.35 - 1e-6
    assert float(smoothed.max()) <= 1.25 + 1e-6
    assert float(smoothed[:, 22].mean()) > 1.20
    assert float(smoothed[:, 25].mean()) < 0.40


def test_crevice_ssao_gate_prevents_double_darkening_and_depth_adapter_quality_tier():
    from relight_engine.adapters.depth_anything_hf import HuggingFaceDepthAnythingV2
    from relight_engine.geometry import refine_depth_to_rgb_edges
    from relight_engine.lighting import evaluate_rig

    adapter_hi = HuggingFaceDepthAnythingV2(quality_tier="studio_high", relative_near_is_high=True)
    assert adapter_hi.model_id == HuggingFaceDepthAnythingV2.HIGH_PRECISION_RELATIVE_MODEL

    h, w = 32, 32
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    depth = np.full((h, w), 2.5, dtype=np.float32)
    depth[:, 14:18] = 2.9
    rgb = np.full((h, w, 3), 0.55, dtype=np.float32)
    rgb[:, 14:18] = 0.20
    snapped = refine_depth_to_rgb_edges(depth, rgb)
    assert float(snapped.min()) >= 2.5 - 1e-5 and float(snapped.max()) <= 2.9 + 1e-5

    pts = backproject_depth(depth, cam)
    normals = np.zeros_like(pts)
    normals[..., 2] = -1.0
    baked_crevice = np.full((h, w), 0.60, dtype=np.float32)
    baked_crevice[:, 14:18] = 0.10
    sc = SceneBuffers(
        rgb_srgb=rgb, depth=depth, depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32), depth_kind="relative",
        normals=normals, geometry_confidence=np.ones((h, w), np.float32),
        source_shading=baked_crevice,
    )
    rig = LightingRig(ambient_strength=0.10, lights=(Light("k", "directional", direction=(0.0, 0.0, -1.0), strength=1.0, casts_shadows=False),))
    d_gated, _, _ = evaluate_rig(sc, cam, pts, normals, rig, EngineSettings(ssao_strength=0.5, preserve_fill_balance=0.55))
    sc_no_sh = SceneBuffers(
        rgb_srgb=rgb, depth=depth, depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32), depth_kind="relative",
        normals=normals, geometry_confidence=np.ones((h, w), np.float32),
    )
    d_ungated, _, _ = evaluate_rig(sc_no_sh, cam, pts, normals, rig, EngineSettings(ssao_strength=0.5, preserve_fill_balance=0.55))
    assert float(d_gated[:, 15].mean()) > float(d_ungated[:, 15].mean()) + 0.02
