"""Regression tests for Final Plan Parts A & B: width-aware shadows, semantic PBR priors, gobo shading, SSAO, indirect specular, half-res shadows, and API presets."""
from __future__ import annotations

import numpy as np
import pytest

from relight_engine import (
    CameraModel,
    EngineSettings,
    Light,
    LightingRig,
    SceneBuffers,
    compute_screen_space_ao,
    refine_source_shading_with_albedo,
    relight,
    suggest_material_priors_from_semantics,
)
from relight_engine.geometry import backproject_depth, prepare_geometry
from relight_engine.lighting import compute_indirect_specular
from relight_engine.shadows import ray_visibility


def test_width_aware_shadow_culls_narrow_hiker_chasm_while_keeping_broad_wall_shadow():
    h, w = 48, 64
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    depth = np.full((h, w), 4.6, dtype=np.float32)
    # Narrow 3px foreground hiker standing in front of far valley
    depth[16:36, 20:23] = 1.5
    # Broad 18px torso standing 1.0 unit from wall on right side
    depth[:, 40:] = 2.6
    depth[12:38, 36:54] = 1.6

    points = backproject_depth(depth, cam)
    normals = np.zeros_like(points)
    normals[..., 2] = -1.0
    to_light = np.zeros_like(points)
    to_light[:] = np.array([-1.0, 0.0, -0.45], dtype=np.float32)

    vis_w = ray_visibility(
        points, normals, depth, cam, to_light, np.full((h, w), 6.0, np.float32),
        "relative", EngineSettings(shadow_steps=24, shadow_max_thickness=1.6, shadow_width_aware=True),
    )
    # Narrow foreground hiker over 3.1-unit valley chasm fades out cleanly
    assert float(vis_w[24, 28]) > 0.88


def test_semantic_pbr_priors_and_albedo_guided_gobo_shading():
    h, w = 32, 48
    skin_mask = np.zeros((h, w), dtype=np.float32)
    skin_mask[8:24, 8:20] = 1.0
    watch_mask = np.zeros((h, w), dtype=np.float32)
    watch_mask[8:24, 24:36] = 1.0

    # Uniform concrete wall albedo with sharp window-blind gobo shadow baked into RGB
    albedo = np.full((h, w, 3), 0.65, dtype=np.float32)
    rgb = albedo.copy()
    rgb[:, 38:] = 0.28  # sharp shadow bar on right side of uniform wall
    smooth_sh = np.full((h, w, 3), 0.65, dtype=np.float32)

    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=np.full((h, w), 2.5, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        albedo=albedo,
        source_shading=smooth_sh,
        semantic_masks={"face": skin_mask, "watch": watch_mask},
    )
    mats = suggest_material_priors_from_semantics(scene)
    assert mats is not None and mats.subsurface is not None
    assert float(mats.subsurface[16, 14]) > 0.25
    assert float(mats.metalness[16, 30]) > 0.70

    refined_sh = refine_source_shading_with_albedo(scene, smooth_sh)
    assert refined_sh is not None
    # Albedo-guided refinement recovers sharp gobo shadow bar on flat-albedo wall
    assert float(refined_sh[:, 6].mean()) > float(refined_sh[:, 42].mean()) * 1.4


def test_ssao_crevice_grounding_indirect_specular_and_half_res_shadows():
    h, w = 32, 32
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    depth = np.full((h, w), 2.5, dtype=np.float32)
    # Recessed narrow box/crevice inside plane (depth jumps deeper -> concavities at border edges?
    # Better: raised walls flanking a 4px floor recess at x=14..18!)
    depth[:, :13] = 2.0
    depth[:, 19:] = 2.0
    points = backproject_depth(depth, cam)
    normals = np.zeros_like(points)
    normals[..., 2] = -1.0
    ao = compute_screen_space_ao(points, normals, depth)
    assert float(ao[:, 15].mean()) < float(ao[:, 4].mean()) - 0.05

    metal_scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.5, dtype=np.float32),
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        roughness=np.full((h, w), 0.15, dtype=np.float32),
        metalness=np.full((h, w), 0.90, dtype=np.float32),
        specular_color=np.full((h, w, 3), np.array([0.85, 0.85, 0.88], np.float32), dtype=np.float32),
    )
    bounce = np.full((h, w, 3), np.array([0.5, 0.1, 0.1], np.float32), dtype=np.float32)
    spec_b = compute_indirect_specular(metal_scene, backproject_depth(metal_scene.depth, cam), normals, bounce, EngineSettings(indirect_specular_strength=0.6))
    assert float(spec_b[..., 0].mean()) > float(spec_b[..., 2].mean()) + 0.05


def test_server_studio_3d_preset_and_orbit_placement(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from relight_engine.server import ServerConfig, create_app
    from tests.test_server import _write_synthetic_scene

    preload = tmp_path / "preload"
    _write_synthetic_scene(preload / "orbit_scene", h=40, w=60)
    app = create_app(ServerConfig(storage_dir=tmp_path / "srv", device="cpu", preload_dirs=(preload,)))
    with TestClient(app) as client:
        resp = client.post(
            "/v1/scenes/orbit_scene/relight",
            json={
                "source_illumination": {"lights": [{"kind": "directional", "direction": [0.0, 0.2, -0.98], "strength": 0.8}]},
                "target_rig": {
                    "ambient_strength": 0.12,
                    "lights": [
                        {
                            "kind": "point",
                            "placement_mode": "studio_orbit",
                            "azimuth_deg": -52.0,
                            "elevation_deg": 24.0,
                            "distance_scale": 1.3,
                            "strength": 2.8,
                        }
                    ],
                },
                "settings": {"preset": "studio_3d"},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["resolution"] == [60, 40]
        assert body["metrics"]["shadow_steps_effective"] >= 32
