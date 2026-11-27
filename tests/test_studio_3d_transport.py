"""Whole-scene 3D studio lighting tests: cross-object VPL radiosity bounce, studio orbit placement, and wall/skin differentiation."""
from __future__ import annotations

import numpy as np

from relight_engine import (
    CameraModel,
    EngineSettings,
    Light,
    LightingRig,
    SceneBuffers,
    classify_scene_layers,
    place_studio_light,
    relight,
)
from relight_engine.calibration import SourceLightCalibration
from relight_engine.geometry import backproject_depth
from relight_engine.lighting import compute_indirect_bounce


def test_cross_depth_3d_surfel_bounce_transfers_color_from_floor_and_wall_to_subject():
    h, w = 48, 64
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    # Wall in back at Z=4.2, bright red floor in lower third Z=2.0..4.0 sloping, neutral subject at Z=1.8
    depth = np.full((h, w), 4.2, dtype=np.float32)
    depth[32:, :] = 2.0
    depth[12:36, 22:42] = 1.8  # neutral center subject

    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0  # wall & front face
    normals[32:, :] = np.array([0.0, 0.85, -0.52], dtype=np.float32)  # upward floor
    normals[24:36, 22:42] = np.array([0.0, -0.45, -0.89], dtype=np.float32)  # downward-facing underside of subject

    albedo = np.full((h, w, 3), 0.55, dtype=np.float32)
    albedo[32:, :] = np.array([0.90, 0.12, 0.10], dtype=np.float32)  # red studio floor

    scene = SceneBuffers(
        rgb_srgb=albedo.copy(),
        depth=depth,
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        albedo=albedo,
    )
    pts = backproject_depth(depth, cam)
    direct = np.zeros((h, w, 3), dtype=np.float32)
    direct[32:, :] = 2.2  # floor strongly lit by studio light

    bounce = compute_indirect_bounce(
        scene, pts, normals, direct,
        EngineSettings(indirect_bounce_strength=0.65, indirect_bounce_radius=24.0),
    )
    # Red studio floor bounces warm red radiance up onto the shadowed underside of the foreground subject (>10 px & across depth jump)
    subj_under = bounce[25:31, 26:38]
    assert float(subj_under[..., 0].mean()) > float(subj_under[..., 2].mean()) + 0.04


def test_place_studio_light_and_whole_room_wall_shadow_with_neutral_wall_color():
    h, w = 48, 64
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    depth = np.full((h, w), 4.2, dtype=np.float32)
    depth[10:42, 24:38] = 1.8  # subject standing in front of concrete wall
    subj_mask = np.zeros((h, w), dtype=np.float32)
    subj_mask[10:42, 24:38] = 1.0
    rgb = np.full((h, w, 3), 0.58, dtype=np.float32)  # neutral concrete wall
    rgb[10:42, 24:38] = np.array([0.12, 0.16, 0.35], dtype=np.float32)  # navy suit

    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=depth,
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        albedo=rgb.copy(),
        source_shading=np.full((h, w), 0.55, dtype=np.float32),
        semantic_masks={"person": subj_mask},
    )
    s_mask, e_mask, sky_mask = classify_scene_layers(scene)
    assert float(s_mask[24, 30]) > 0.8 and float(e_mask[24, 8]) > 0.8 and float(sky_mask.max()) < 0.1

    pos, aim = place_studio_light(scene, cam, azimuth_deg=-52.0, elevation_deg=24.0, distance_scale=1.3)
    assert pos[0] < 0.0 and pos[1] > 0.0 and pos[2] < 4.2

    src = SourceLightCalibration(key_direction=(0.0, 0.2, -0.98), key_strength=0.8, ambient_strength=0.35).to_rig()
    tgt = LightingRig(
        ambient_strength=0.14,
        lights=(Light("studio_key", "point", position=pos, direction=aim, color=(1.0, 0.65, 0.35), strength=3.2, radius=0.30, casts_shadows=True),),
    )
    res = relight(scene, cam, src, tgt, EngineSettings.studio_3d())
    # Left lit wall catches warm studio light while shadowed right wall receives cast shadow from the subject!
    lit_wall = res.rgb_srgb[20:32, 6:16]
    shad_wall = res.rgb_srgb[20:32, 42:54]
    assert float(lit_wall.mean()) > float(shad_wall.mean()) + 0.08
    # Neutral concrete wall visibly absorbs warm hue shift (R > B)
    assert float(lit_wall[..., 0].mean()) > float(lit_wall[..., 2].mean()) + 0.06
    # Navy subject retains blue dominance (B > R) instead of mutating to purple/orange
    assert float(res.rgb_srgb[20:32, 26:36, 2].mean()) > float(res.rgb_srgb[20:32, 26:36, 0].mean())
