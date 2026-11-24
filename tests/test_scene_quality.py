from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from relight_engine.contracts import SceneBuffers
from relight_engine.quality import evaluate_scene_buffers


def make_scene(*, flat_depth=False, include_masks=True):
    h, w = 24, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = np.full((h, w), 2.0, np.float32) if flat_depth else (1.0 + xx / w + yy / (2 * h)).astype(np.float32)
    rgb = np.stack([xx / max(w - 1, 1), yy / max(h - 1, 1), np.full((h, w), 0.4)], axis=-1).astype(np.float32)
    confidence = np.full((h, w), 0.8, np.float32)
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = -1.0
    geometry_confidence = np.full((h, w), 0.75, np.float32)
    label_mask = np.zeros((h, w), np.float32)
    label_mask[3:20, 5:27] = 0.9
    scene = SceneBuffers(
        rgb_srgb=rgb, depth=depth, depth_confidence=confidence,
        lightable_mask=np.ones((h, w), np.float32), depth_kind="relative",
        normals=normals, geometry_confidence=geometry_confidence,
        semantic_masks={"sample_object": label_mask} if include_masks else {},
        instance_masks={"sample_object__001": label_mask} if include_masks else {},
        metadata={"material_maps_estimated": False},
    )
    return scene


def test_quality_report_marks_scene_structurally_valid_without_claiming_accuracy():
    report = evaluate_scene_buffers(make_scene())
    assert report["status"] == "structurally_valid"
    assert report["assessment_scope"] == "structural_and_degeneracy_checks_only_not_ground_truth_quality"
    assert report["image"]["depth_kind"] == "relative"
    assert report["depth"]["valid_fraction_of_visible_pixels"] == pytest.approx(1.0)
    assert report["geometry"]["normals"]["available"] is True
    assert report["masks"]["semantic_class_count"] == 1
    assert report["masks"]["instance_count"] == 1
    assert report["materials"]["material_estimation_claimed"] is False


def test_quality_report_warns_about_flat_depth_map():
    report = evaluate_scene_buffers(make_scene(flat_depth=True))
    assert report["status"] == "structurally_valid_with_warnings"
    assert any("flat" in warning.lower() for warning in report["warnings"])


def test_quality_report_warns_when_normals_or_masks_are_missing():
    scene = make_scene(include_masks=False)
    scene.normals = None
    report = evaluate_scene_buffers(scene)
    assert any("normals are absent" in warning.lower() for warning in report["warnings"])
    assert any("no semantic class masks" in warning.lower() for warning in report["warnings"])


def test_quality_report_detects_near_duplicate_instances():
    scene = make_scene()
    scene.instance_masks["sample_object__002"] = scene.instance_masks["sample_object__001"].copy()
    report = evaluate_scene_buffers(scene)
    assert len(report["masks"]["near_duplicate_instances"]) == 1
    assert any("near-identical" in warning for warning in report["warnings"])


def test_quality_report_excludes_fully_transparent_pixels_from_depth_coverage():
    scene = make_scene()
    alpha = np.ones(scene.shape, np.float32)
    alpha[:, :8] = 0
    scene.alpha_mask = alpha
    scene.depth[:, :8] = 0
    report = evaluate_scene_buffers(scene)
    assert report["image"]["visible_fraction"] == pytest.approx(0.75)
    assert report["depth"]["valid_fraction_of_visible_pixels"] == pytest.approx(1.0)


def test_shared_renderer_is_independent_of_scene_domain_metadata():
    from relight_engine.contracts import CameraModel, LightingRig, Light, EngineSettings
    from relight_engine.engine import relight

    base = make_scene()
    camera = CameraModel(fx=35.0, fy=35.0, cx=16.0, cy=12.0)
    source = LightingRig(ambient_strength=0.2, lights=(
        Light(name="source", kind="directional", direction=(0.0, 0.0, -1.0), strength=0.8),
    ))
    target = LightingRig(ambient_strength=0.15, lights=(
        Light(name="target", kind="directional", direction=(0.6, 0.0, -0.8), strength=1.0,
              casts_shadows=False),
    ))
    portrait_scene = make_scene()
    portrait_scene.metadata["scene_domain"] = "portrait"
    vehicle_scene = make_scene()
    vehicle_scene.metadata["scene_domain"] = "vehicle"
    result_a = relight(portrait_scene, camera, source, target, EngineSettings(shadow_steps=4))
    result_b = relight(vehicle_scene, camera, source, target, EngineSettings(shadow_steps=4))
    assert np.allclose(result_a.rgb_srgb, result_b.rgb_srgb, atol=0, rtol=0)



def test_material_report_records_map_provenance_confidence_and_supported_families():
    scene = make_scene()
    scene.roughness = np.full(scene.shape, 0.32, dtype=np.float32)
    scene.metalness = np.full(scene.shape, 0.85, dtype=np.float32)
    scene.specular_color = np.broadcast_to(
        np.array([0.78, 0.24, 0.12], dtype=np.float32),
        (*scene.shape, 3),
    ).copy()
    scene.material_confidence = np.full(scene.shape, 0.72, dtype=np.float32)
    scene.material_metadata = {
        "roughness": {"source": "user_authored", "calibrated": False},
        "metalness": {"source": "user_authored", "calibrated": False},
        "specular_color": {"source": "user_authored", "calibrated": False},
        "material_confidence": {"source": "user_reviewed", "calibrated": False},
    }

    report = evaluate_scene_buffers(scene)
    materials = report["materials"]

    assert materials["maps"]["roughness"]["source"] == "user_authored"
    assert materials["maps"]["specular_color"]["available"] is True
    assert materials["maps"]["specular_color"]["expected_shape"] == "HxWx3_linear_RGB_conductor_F0"
    assert materials["material_confidence"]["available"] is True
    assert materials["material_confidence"]["mean"] == pytest.approx(0.72)
    assert "ggx_conductor_with_RGB_F0" in materials["implemented_response_families"]
    assert "transmission_and_refraction" in materials["not_implemented_response_families"]
    assert materials["material_estimation_claimed"] is False


def test_material_report_marks_missing_maps_as_conservative_priors():
    scene = make_scene()
    report = evaluate_scene_buffers(scene)
    materials = report["materials"]

    assert materials["maps"]["roughness"]["source"] == "conservative_prior"
    assert materials["maps"]["roughness"]["fallback_prior"] == pytest.approx(0.75)
    assert materials["maps"]["metalness"]["fallback_prior"] == pytest.approx(0.0)
    assert materials["maps"]["specular_level"]["fallback_prior"] == pytest.approx(0.04)
    assert materials["material_confidence"]["available"] is False
    assert materials["material_estimation_claimed"] is False


def test_material_report_warns_for_unsourced_or_unconfident_estimated_maps():
    scene = make_scene()
    scene.roughness = np.full(scene.shape, 0.4, dtype=np.float32)
    scene.material_metadata = {
        "roughness": {"source": "model_estimated"}
    }
    report = evaluate_scene_buffers(scene)

    assert report["materials"]["maps"]["roughness"]["source"] == "model_estimated"
    assert any("no per-pixel material-confidence" in w.lower() for w in report["warnings"])

    scene.material_metadata = {}
    report_without_source = evaluate_scene_buffers(scene)
    assert any("lack declared provenance" in w.lower() for w in report_without_source["warnings"])


def test_quality_report_records_intrinsic_availability_and_provenance():
    scene = make_scene()
    scene.albedo = np.full((*scene.shape, 3), 0.5, dtype=np.float32)
    scene.source_shading = np.full(scene.shape, 0.7, dtype=np.float32)
    scene.intrinsic_metadata = {"source": "test_intrinsic_estimator", "calibrated": False}
    report = evaluate_scene_buffers(scene)
    assert report["intrinsics"]["albedo_available"] is True
    assert report["intrinsics"]["source_shading_available"] is True
    assert report["intrinsics"]["metadata"]["source"] == "test_intrinsic_estimator"
