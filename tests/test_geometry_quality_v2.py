from __future__ import annotations

import numpy as np
import pytest

from relight_engine.contracts import (
    CameraModel,
    EngineSettings,
    Light,
    LightingRig,
    SceneBuffers,
)
from relight_engine.engine import relight
from relight_engine.geometry import backproject_depth, estimate_normals
from relight_engine.quality import evaluate_scene_buffers


def make_scene(
    *,
    depth_kind: str = "relative",
    confidence_value: float = 1.0,
    stored_normals: bool = True,
) -> SceneBuffers:
    h, w = 32, 40
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    rgb = np.full((h, w, 3), 0.4, dtype=np.float32)
    confidence = np.full((h, w), confidence_value, dtype=np.float32)
    normals = None
    if stored_normals:
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[..., 2] = -1.0
    geometry_confidence = np.full((h, w), confidence_value, dtype=np.float32)
    object_mask = np.zeros((h, w), dtype=np.float32)
    object_mask[5:27, 8:32] = 0.9
    return SceneBuffers(
        rgb_srgb=rgb,
        depth=depth,
        depth_confidence=confidence,
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind=depth_kind,
        normals=normals,
        geometry_confidence=geometry_confidence,
        semantic_masks={"test_object": object_mask},
        instance_masks={"test_object__001": object_mask.copy()},
        metadata={
            "depth": {"confidence_source": "synthetic_test_fixture"},
            "geometry_confidence_source": "synthetic_test_fixture",
            "material_maps_estimated": False,
        },
    )


def test_edge_aware_normals_do_not_mix_across_a_depth_step():
    h, w = 32, 48
    depth = np.full((h, w), 2.0, dtype=np.float32)
    depth[:, w // 2 :] = 6.0
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    points = backproject_depth(depth, camera)

    normals, reliability = estimate_normals(
        points, depth, np.ones_like(depth)
    )

    assert np.isfinite(normals).all()
    assert np.isfinite(reliability).all()
    assert np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-5)
    edge_band = reliability[:, w // 2 - 1 : w // 2 + 1]
    away_from_edge = np.concatenate(
        [reliability[:, : w // 2 - 5], reliability[:, w // 2 + 5 :]],
        axis=1,
    )
    assert float(np.mean(edge_band)) < float(np.mean(away_from_edge)) * 0.1


def test_invalid_depth_pixels_receive_fallback_normals_and_zero_reliability():
    depth = np.full((12, 16), 2.0, dtype=np.float32)
    depth[4:8, 6:10] = 0.0
    camera = CameraModel(30.0, 30.0, 8.0, 6.0)
    points = backproject_depth(depth, camera)
    normals, reliability = estimate_normals(
        points, depth, np.ones_like(depth)
    )

    assert np.all(reliability[4:8, 6:10] == 0.0)
    assert reliability[3, 7] < reliability[2, 7]
    assert np.all(normals[4:8, 6:10] == np.array([0.0, 0.0, -1.0]))
    assert np.isfinite(normals).all()


def test_normal_reliability_is_not_derived_from_cross_product_magnitude():
    depth = np.full((24, 32), 2.0, dtype=np.float32)
    confidence = np.full_like(depth, 0.7)
    camera_a = CameraModel(40.0, 40.0, 16.0, 12.0)
    camera_b = CameraModel(400.0, 400.0, 16.0, 12.0)

    normals_a, reliability_a = estimate_normals(
        backproject_depth(depth, camera_a), depth, confidence
    )
    normals_b, reliability_b = estimate_normals(
        backproject_depth(depth, camera_b), depth, confidence
    )

    assert np.allclose(normals_a, normals_b, atol=1e-5)
    assert np.allclose(reliability_a, reliability_b, atol=1e-5)
    assert float(np.mean(reliability_a)) == pytest.approx(0.7, abs=1e-5)


def test_estimate_normals_rejects_misaligned_confidence():
    depth = np.ones((8, 8), dtype=np.float32)
    points = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="depth_confidence"):
        estimate_normals(points, depth, np.ones((7, 8), dtype=np.float32))


def test_quality_report_identifies_constant_prior_as_uncalibrated():
    scene = make_scene(confidence_value=0.45)
    scene.metadata["depth"]["confidence_source"] = (
        "conservative constant prior; model does not emit calibrated confidence"
    )
    report = evaluate_scene_buffers(scene)

    provenance = report["confidence_provenance"]["depth"]
    assert provenance["kind"] == "constant_prior"
    assert provenance["calibrated"] is False
    assert any("constant prior" in warning.lower() for warning in report["warnings"])


def test_relative_depth_can_pass_normal_gate_but_not_metric_distance_gate():
    relative_scene = make_scene(depth_kind="relative")
    relative_report = evaluate_scene_buffers(relative_scene)

    assert relative_report["lighting_readiness"]["normal_dependent_shading"][
        "structural_gate_passed"
    ] is True
    assert relative_report["lighting_readiness"]["metric_distance_falloff"][
        "metric_depth_available"
    ] is False

    metric_scene = make_scene(depth_kind="metric")
    metric_report = evaluate_scene_buffers(metric_scene)
    assert metric_report["lighting_readiness"]["metric_distance_falloff"][
        "metric_depth_available"
    ] is True


def test_strict_geometry_gate_accepts_reliable_on_demand_normals():
    scene = make_scene(stored_normals=False, depth_kind="metric")
    camera = CameraModel(40.0, 40.0, 20.0, 16.0)
    rig = LightingRig(
        ambient_strength=0.15,
        lights=(
            Light(
                "key",
                "directional",
                strength=0.7,
                direction=(0.0, 0.0, -1.0),
                casts_shadows=False,
            ),
        ),
    )

    result = relight(
        scene,
        camera,
        rig,
        rig,
        EngineSettings(shadow_steps=4),
        require_geometry_ready=True,
    )

    assert result.metrics["geometry_readiness_gate_enforced"] is True
    assert result.metrics["geometry_readiness_gate"]["structural_gate_passed"] is True


def test_strict_geometry_gate_blocks_low_confidence_geometry():
    scene = make_scene(
        stored_normals=False,
        depth_kind="metric",
        confidence_value=0.05,
    )
    camera = CameraModel(40.0, 40.0, 20.0, 16.0)
    rig = LightingRig(ambient_strength=0.15)

    with pytest.raises(ValueError, match="structural gate"):
        relight(
            scene,
            camera,
            rig,
            rig,
            EngineSettings(shadow_steps=4),
            require_geometry_ready=True,
        )
