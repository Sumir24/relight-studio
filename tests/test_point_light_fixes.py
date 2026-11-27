"""Regression tests for point-light localization, front-hemisphere source fit, sky protection, and highlight/halftone safety."""
from __future__ import annotations

import numpy as np

from relight_engine import (
    CameraModel,
    EngineSettings,
    Light,
    LightingRig,
    SceneBuffers,
    detect_sky_horizon_mask,
    estimate_source_illumination,
    place_point_light_at_uv,
    relight,
)
from relight_engine.calibration import SourceLightCalibration


def test_estimate_source_illumination_stays_front_hemisphere_on_flat_wall_and_ground_dominant_scene():
    h, w = 40, 48
    # Portrait-like vertical wall (nz = -1) with subtle horizontal gradient
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shading = np.clip(0.35 + 0.30 * (xx / (w - 1)), 0.1, 1.0)[..., None].repeat(3, axis=-1).astype(np.float32)
    scene_wall = SceneBuffers(
        rgb_srgb=np.clip(0.6 * shading, 0.0, 1.0),
        depth=np.full((h, w), 4.5, np.float32),
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), np.float32),
        albedo=np.full((h, w, 3), 0.6, np.float32),
        source_shading=shading,
    )
    si_wall = estimate_source_illumination(scene_wall, normals)
    assert len(si_wall.lights) >= 1
    assert si_wall.lights[0].direction[2] <= -0.15

    # Ground-dominated scene (80% flat upward ground + 20% vertical subject facing camera)
    normals_g = np.zeros((h, w, 3), dtype=np.float32)
    normals_g[..., 1] = 0.90
    normals_g[..., 2] = -0.436
    normals_g[10:30, 18:30] = np.array([0.0, -0.1, -0.995], dtype=np.float32)
    sh_g = np.full((h, w, 3), 0.5, dtype=np.float32)
    scene_g = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.4, np.float32),
        depth=np.full((h, w), 2.3, np.float32),
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
        normals=normals_g,
        geometry_confidence=np.ones((h, w), np.float32),
        albedo=np.full((h, w, 3), 0.45, np.float32),
        source_shading=sh_g,
    )
    si_g = estimate_source_illumination(scene_g, normals_g)
    assert si_g.lights[0].direction[2] <= -0.15


def test_place_point_light_at_uv_clamps_in_front_of_foreground_subject():
    h, w = 48, 64
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    # Background asphalt/wall at Z=4.8, foreground person at center Z=1.8
    depth = np.full((h, w), 4.8, dtype=np.float32)
    person_mask = np.zeros((h, w), dtype=np.float32)
    depth[10:42, 22:42] = 1.8
    person_mask[10:42, 22:42] = 1.0
    scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.4, np.float32),
        depth=depth,
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
        semantic_masks={"person": person_mask},
    )
    # Even when picking upper-left background corner (u=0.1, v=0.2), Z must stay ahead of foreground Z=1.8!
    lx, ly, lz = place_point_light_at_uv(scene, cam, 0.10, 0.20)
    assert lz < 1.8
    assert lx < 0.0 and ly > 0.0


def test_sky_horizon_mask_blocks_mountain_shadow_halos_and_sky_tinting():
    h, w = 48, 64
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    depth = np.full((h, w), 5.0, dtype=np.float32)
    depth[24:, :] = 2.0  # lower foreground ridge
    depth[16:28, 28:36] = 1.6  # mountain peak sticking into sky
    sky_mask_in = np.zeros((h, w), dtype=np.float32)
    sky_mask_in[:24, :] = 1.0
    sky_mask_in[16:28, 28:36] = 0.0

    rgb = np.full((h, w, 3), 0.35, dtype=np.float32)
    rgb[:24, :] = np.array([0.45, 0.65, 0.92], dtype=np.float32)
    rgb[16:28, 28:36] = 0.25

    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=depth,
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), np.float32),
        semantic_masks={"clear_blue_sky": sky_mask_in},
    )
    detected_sky = detect_sky_horizon_mask(scene)
    assert float(detected_sky[4, 10]) > 0.9

    src = SourceLightCalibration(key_direction=(0.0, 0.2, -0.98), key_strength=0.8, ambient_strength=0.3).to_rig()
    pt_pos = place_point_light_at_uv(scene, cam, 0.85, 0.45)
    tgt = LightingRig(
        ambient_strength=0.1,
        lights=(Light("red_pt", "point", position=pt_pos, color=(1.0, 0.15, 0.10), strength=3.0, casts_shadows=True),),
    )
    res = relight(scene, cam, src, tgt, EngineSettings(protect_sky=True))
    # Clear sky pixels keep exact original sky RGB and no peak shadow halo!
    assert np.allclose(res.rgb_srgb[2:12, 6:22], scene.rgb_srgb[2:12, 6:22], atol=1e-4)
    assert float(res.shadow_visibility[2:12, 20:27].min()) > 0.95


def test_no_albedo_low_confidence_soft_knee_prevents_bright_stipple_blowout():
    h, w = 32, 32
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    rgb = np.full((h, w, 3), 0.72, dtype=np.float32)
    rgb[:, :8] = 0.05  # black ink border
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=np.full((h, w), 2.1, np.float32),
        depth_confidence=np.full((h, w), 0.45, np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.full((h, w), 0.45, np.float32),
    )
    # Weak source estimate (like halftone art without albedo) + strong close point light
    src = LightingRig(ambient_strength=0.05, lights=(Light("w", "directional", direction=(0.0, 0.0, -1.0), strength=0.14, casts_shadows=False),))
    tgt = LightingRig(ambient_strength=0.10, lights=(Light("p", "point", position=(0.0, 0.0, 1.3), strength=3.0, casts_shadows=False),))
    res = relight(scene, cam, src, tgt, EngineSettings())
    # Soft-knee + source-diffuse floor keep bright midtones below hard 1.0 flat clipping while ink stays dark
    assert float(res.rgb_srgb[16, 20].mean()) < 0.97
    assert float(res.rgb_srgb[16, 4].mean()) < 0.25

    res_pres = relight(scene, cam, src, tgt, EngineSettings.real_preserving())
    assert float(res_pres.rgb_srgb[16, 20].mean()) < 0.97
    assert float(res_pres.rgb_srgb[16, 4].mean()) < float(res.rgb_srgb[16, 4].mean())
