from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from relight_engine.contracts import CameraModel, EngineSettings, Light, LightingRig, SceneBuffers


def make_scene() -> SceneBuffers:
    h, w = 6, 8
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = -1.0
    return SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.4, np.float32),
        depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), np.float32),
    )


def test_light_rejects_malformed_direction_vector():
    light = Light("bad-direction", "directional", direction=(1.0, 0.0))
    with pytest.raises(ValueError):
        light.validate()


def test_light_rejects_non_finite_strength():
    light = Light("nan-strength", "directional", strength=float("nan"), direction=(0, 0, -1))
    with pytest.raises(ValueError):
        light.validate()


def test_lighting_rig_rejects_non_finite_ambient_strength():
    with pytest.raises(ValueError):
        LightingRig(ambient_strength=float("nan")).validate()


def test_engine_settings_reject_non_finite_values():
    with pytest.raises(ValueError):
        EngineSettings(base_blur_sigma=float("nan")).validate()


def test_scene_buffers_reject_out_of_range_geometry_confidence():
    scene = make_scene()
    scene.geometry_confidence[0, 0] = 1.2
    with pytest.raises(ValueError):
        scene.validate()


def test_scene_buffers_reject_zero_normal_on_valid_depth_pixel():
    scene = make_scene()
    scene.normals[0, 0] = 0
    with pytest.raises(ValueError):
        scene.validate()


def test_scene_buffers_accept_zero_normal_where_depth_is_invalid():
    scene = make_scene()
    scene.depth[0, 0] = 0
    scene.normals[0, 0] = 0
    scene.validate()


def test_camera_intrinsics_reject_non_positive_focal_length():
    with pytest.raises(ValueError):
        CameraModel(0.0, 10.0, 4.0, 3.0).validate()


def test_scene_buffers_validate_dynamic_semantic_and_instance_maps():
    scene = make_scene()
    region = np.zeros(scene.shape, np.float32)
    region[1:3, 2:5] = 0.8
    scene.semantic_masks = {"ceramic_mug": region}
    scene.instance_masks = {"ceramic_mug__001": region.copy()}
    scene.validate()
    scene.instance_masks["ceramic_mug__001"][0, 0] = 1.2
    with pytest.raises(ValueError, match="instance mask"):
        scene.validate()


def test_scene_buffers_validates_intrinsic_maps_and_rejects_bad_shapes_or_negatives():
    scene = make_scene()
    scene.albedo = np.full((*scene.shape, 3), 0.5, dtype=np.float32)
    scene.source_shading = np.full(scene.shape, 0.8, dtype=np.float32)
    scene.non_diffuse_residual = np.zeros((*scene.shape, 3), dtype=np.float32)
    scene.validate()

    bad = make_scene()
    bad.albedo = np.full((*scene.shape, 4), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="albedo"):
        bad.validate()

    bad_shading = make_scene()
    bad_shading.source_shading = np.full(scene.shape, -0.1, dtype=np.float32)
    with pytest.raises(ValueError, match="source_shading"):
        bad_shading.validate()


def test_engine_settings_validates_renderer_mode_and_detail_lock_strength():
    defaults = EngineSettings()
    assert defaults.color_ratio_min <= 0.25 and defaults.color_ratio_max >= 3.0
    EngineSettings(renderer_mode="intrinsic", detail_lock_strength=0.9, preserve_source_lighting=True).validate()
    with pytest.raises(ValueError, match="renderer_mode"):
        EngineSettings(renderer_mode="unknown").validate()
    with pytest.raises(ValueError, match="detail_lock_strength"):
        EngineSettings(detail_lock_strength=1.5).validate()
    with pytest.raises(ValueError, match="preserve_source_lighting"):
        EngineSettings(preserve_source_lighting="yes").validate()
