from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from relight_engine.workflows import source_calibration_from_dict, target_rig_from_dict, apply_user_material_assignments
from relight_engine.contracts import SceneBuffers


def test_source_calibration_requires_explicit_direction():
    with pytest.raises(ValueError):
        source_calibration_from_dict({"key_strength": 0.8})


def test_target_rig_builder_supports_temperature_and_directional_light():
    rig = target_rig_from_dict({
        "ambient_strength": 0.1,
        "ambient_temperature_kelvin": 6500,
        "ambient_ground_temperature_kelvin": 3200,
        "lights": [{"kind": "directional", "direction": [1, 1, -1], "color_temperature_kelvin": 5000, "wrap": 0.35}],
    })
    assert len(rig.lights) == 1
    assert rig.lights[0].wrap == pytest.approx(0.35)
    assert rig.ambient_ground_color is not None
    rig.validate()


def test_target_rig_rejects_conflicting_color_inputs():
    with pytest.raises(ValueError):
        target_rig_from_dict({"lights": [{"kind": "directional", "direction": [0, 1, -1],
                                            "color": [1, 1, 1], "color_temperature_kelvin": 5000}]})


def test_user_material_assignment_uses_scene_mask_and_records_manual_origin():
    h, w = 6, 8
    mask = np.zeros((h, w), np.float32); mask[2:4, 2:6] = 1
    scene = SceneBuffers(
        rgb_srgb=np.zeros((h, w, 3), np.float32), depth=np.ones((h, w), np.float32),
        depth_confidence=np.ones((h, w), np.float32), lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
    )
    metadata = apply_user_material_assignments(scene, {"mask_surface_panel": mask}, [{
        "name": "surface panel", "mask_key": "surface_panel", "roughness": 0.9,
    }])
    assert np.allclose(scene.roughness[2:4, 2:6], 0.9)
    assert metadata["neural_material_estimation_used"] is False


def test_material_assignment_requires_mask_from_registered_scene():
    scene = SceneBuffers(
        rgb_srgb=np.zeros((6, 8, 3), np.float32), depth=np.ones((6, 8), np.float32),
        depth_confidence=np.ones((6, 8), np.float32), lightable_mask=np.ones((6, 8), np.float32),
        depth_kind="relative",
    )
    with pytest.raises(ValueError):
        apply_user_material_assignments(scene, {}, [{"name": "glass_region", "mask_key": "glass", "roughness": 0.1}])


def test_target_rig_rejects_unknown_top_level_fields():
    with pytest.raises(ValueError):
        target_rig_from_dict({"ambient_strenght": 0.2})


def test_full_scene_to_relight_workflow_on_synthetic_fixture(tmp_path):
    import json
    from relight_engine.workflows import run_scene_relight

    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    h, w = 24, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([
        np.full((h, w), 0.42, np.float32),
        0.38 + 0.08 * (xx / max(w - 1, 1)),
        0.30 + 0.05 * (yy / max(h - 1, 1)),
    ], axis=-1).astype(np.float32)
    depth = np.full((h, w), 2.0, np.float32)
    conf = np.ones((h, w), np.float32)
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = -1.0
    lightable = np.ones((h, w), np.float32)
    alpha = np.ones((h, w), np.float32); alpha[:, :4] = 0.0
    surface_a = np.zeros((h, w), np.float32); surface_a[:12] = 1.0
    surface_b = np.zeros((h, w), np.float32); surface_b[12:] = 1.0
    person_1 = np.zeros((h, w), np.float32); person_1[2:10, 3:8] = 0.9
    protected = np.zeros((h, w), np.float32); protected[:2] = 1.0
    np.savez_compressed(
        scene_dir / "scene_buffers.npz",
        rgb_srgb=rgb, depth=depth, depth_confidence=conf,
        geometry_confidence=conf, normals=normals, lightable_mask=lightable,
        mask_surface_panel=surface_a, mask_fabric=surface_b,
        mask_instance_person__001=person_1, alpha_mask=alpha, protected_mask=protected,
        roughness=np.full((h, w), 0.42, np.float32),
        metalness=np.full((h, w), 0.25, np.float32),
        specular_level=np.full((h, w), 0.06, np.float32),
        specular_color=np.broadcast_to(
            np.array([0.72, 0.38, 0.18], np.float32), (h, w, 3)
        ).copy(),
        material_confidence=np.full((h, w), 0.6, np.float32),
    )
    manifest = {
        "material_metadata": {
            "roughness": {"source": "user_authored", "calibrated": False},
            "metalness": {"source": "user_authored", "calibrated": False},
            "specular_level": {"source": "user_authored", "calibrated": False},
            "specular_color": {"source": "user_authored", "calibrated": False},
            "material_confidence": {"source": "user_reviewed", "calibrated": False},
        },
        "depth": {"depth_kind": "relative", "model_id": "synthetic-test-fixture"},
        "geometry_confidence": {"source": "synthetic-test-heuristic", "calibrated": False},
        "segmentation": {"model_id": "synthetic-test-fixture"},
        "camera": {"fx": 30.0, "fy": 30.0, "cx": w / 2, "cy": h / 2},
    }
    (scene_dir / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    from relight_engine.workflows import load_scene_artifacts
    loaded_scene, _, _, _ = load_scene_artifacts(scene_dir)
    assert sorted(loaded_scene.semantic_masks) == ["fabric", "surface_panel"]
    assert sorted(loaded_scene.instance_masks) == ["person__001"]
    assert loaded_scene.alpha_mask is not None
    assert loaded_scene.protected_mask is not None
    assert loaded_scene.specular_color is not None
    assert np.allclose(loaded_scene.specular_color[0, 0], [0.72, 0.38, 0.18])
    assert loaded_scene.material_confidence[0, 0] == pytest.approx(0.6)
    assert loaded_scene.material_metadata["specular_color"]["source"] == "user_authored"
    assert loaded_scene.metadata["geometry_confidence_source"] == "synthetic-test-heuristic"
    assert np.all(loaded_scene.protected_mask[:2] == 1.0)

    config = {
        "approved_for_test": True,
        "source_calibration": {
            "key_direction": [0.0, 0.0, -1.0],
            "key_strength": 0.8,
            "key_temperature_kelvin": 5600,
            "ambient_strength": 0.1,
        },
        "target_rig": {
            "ambient_strength": 0.1,
            "lights": [{
                "kind": "directional", "direction": [0.7, 0.0, -0.7],
                "strength": 0.8, "color_temperature_kelvin": 5000,
                "casts_shadows": False,
            }],
        },
        "material_assignments": [{
            "name": "panel_test_material", "mask_key": "surface_panel",
            "roughness": 0.9, "metalness": 1.0, "specular_level": 0.04,
            "specular_color": [0.78, 0.32, 0.14], "confidence": 0.85,
        }],
        "settings": {"base_blur_sigma": 3.0, "shadow_steps": 4},
    }
    config_path = tmp_path / "calibration.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "relit_test.png"
    report = run_scene_relight(scene_dir, config_path, output)

    assert output.is_file()
    from PIL import Image
    with Image.open(output) as rendered:
        assert rendered.mode == "RGBA"
        rendered_alpha = np.asarray(rendered)[..., 3] / 255.0
        assert np.all(rendered_alpha[:, :4] == 0.0)
        assert np.all(rendered_alpha[:, 4:] == 1.0)
    assert Path(str(output) + ".report.json").is_file()
    assert report["status"] == "rendered_with_explicit_user_calibration"
    assert report["source_calibration"]["source"] == "user_calibration_not_automatic_estimation"
    assert report["material_maps"]["neural_material_estimation_used"] is False
    assert report["engine_metrics"]["material_response"]["maps"]["specular_color"]["available"] is True
    assert report["engine_metrics"]["material_response"]["material_confidence"]["available"] is True
    assert report["engine_metrics"]["material_response"]["maps"]["specular_color"]["source"] == "user_assignment_plus_renderer_prior"
    assert report["engine_metrics"]["mean_abs_srgb_change_active"] > 0
    assert report["physical_accuracy_notes"]



def test_user_material_assignment_applies_rgb_f0_confidence_and_provenance():
    h, w = 6, 8
    mask = np.zeros((h, w), np.float32)
    mask[1:5, 2:6] = 1.0
    scene = SceneBuffers(
        rgb_srgb=np.zeros((h, w, 3), np.float32),
        depth=np.ones((h, w), np.float32),
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
    )
    metadata = apply_user_material_assignments(
        scene,
        {"mask_metal_trim": mask},
        [{
            "name": "metal trim",
            "mask_key": "metal_trim",
            "roughness": 0.18,
            "metalness": 1.0,
            "specular_level": 0.04,
            "specular_color": [0.88, 0.55, 0.25],
            "confidence": 0.7,
        }],
    )

    assert scene.specular_color is not None
    assert np.allclose(scene.specular_color[2, 3], [0.88, 0.55, 0.25])
    assert scene.material_confidence[2, 3] == pytest.approx(0.7)
    assert scene.material_confidence[0, 0] == pytest.approx(0.0)
    assert scene.material_metadata["specular_color"]["source"] == "user_assignment_plus_renderer_prior"
    assert "unassigned pixels retain" in scene.material_metadata["specular_color"]["notes"]
    assert scene.material_metadata["roughness"]["source"] == "user_assignment_plus_renderer_prior"
    assert scene.material_metadata["material_confidence"]["calibrated"] is False
    assert "not a calibrated probability" in scene.material_metadata["material_confidence"]["notes"].lower()
    assert metadata["neural_material_estimation_used"] is False


def test_user_material_assignment_resolves_instance_mask_filename_stem():
    h, w = 6, 8
    mask = np.zeros((h, w), np.float32)
    mask[1:4, 2:6] = 1.0
    scene = SceneBuffers(
        rgb_srgb=np.zeros((h, w, 3), np.float32),
        depth=np.ones((h, w), np.float32),
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
    )
    apply_user_material_assignments(
        scene,
        {"mask_instance_person__001": mask},
        [{
            "name": "person_one",
            "mask_key": "instance_mask_person__001.png",
            "roughness": 0.62,
        }],
    )
    assert np.allclose(scene.roughness[1:4, 2:6], 0.62)


def test_run_scene_relight_enforces_structural_geometry_gate_by_default(tmp_path):
    import json
    from relight_engine.workflows import run_scene_relight

    scene_dir = tmp_path / "low_conf_scene"
    scene_dir.mkdir()
    h, w = 16, 20
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = -1.0
    np.savez_compressed(
        scene_dir / "scene_buffers.npz",
        rgb_srgb=np.full((h, w, 3), 0.4, np.float32),
        depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.full((h, w), 0.05, np.float32),
        geometry_confidence=np.full((h, w), 0.05, np.float32),
        normals=normals,
        lightable_mask=np.ones((h, w), np.float32),
    )
    manifest = {
        "depth": {"depth_kind": "relative"},
        "geometry_confidence": {"source": "test", "calibrated": False},
        "camera": {"fx": 20.0, "fy": 20.0, "cx": w / 2, "cy": h / 2},
    }
    (scene_dir / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "approved_for_test": True,
        "source_calibration": {"key_direction": [0.0, 0.0, -1.0]},
        "target_rig": {"lights": [{"kind": "directional", "direction": [0.5, 0.0, -0.8]}]},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="structural gate"):
        run_scene_relight(scene_dir, config_path, tmp_path / "blocked.png")

    config["require_geometry_ready"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    report = run_scene_relight(scene_dir, config_path, tmp_path / "override.png")
    assert report["engine_metrics"]["geometry_readiness_gate_enforced"] is False


def test_load_scene_artifacts_preserves_neural_scene_estimation_flag(tmp_path):
    import json
    from relight_engine.workflows import load_scene_artifacts

    scene_dir = tmp_path / "neural_scene"
    scene_dir.mkdir()
    h, w = 8, 10
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = -1.0
    np.savez_compressed(
        scene_dir / "scene_buffers.npz",
        rgb_srgb=np.full((h, w, 3), 0.4, np.float32),
        depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.full((h, w), 0.45, np.float32),
        geometry_confidence=np.full((h, w), 0.45, np.float32),
        normals=normals,
        lightable_mask=np.ones((h, w), np.float32),
    )
    manifest = {
        "depth": {"depth_kind": "relative", "model_id": "depth-anything/Depth-Anything-V2-Small-hf"},
        "segmentation": {"detector_id": "IDEA-Research/grounding-dino-tiny"},
        "geometry_confidence": {"source": "heuristic", "calibrated": False},
        "camera": {"fx": 10.0, "fy": 10.0, "cx": 5.0, "cy": 4.0},
    }
    (scene_dir / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    scene, _, _, _ = load_scene_artifacts(scene_dir)
    assert scene.metadata["neural_scene_estimation_used"] is True


def test_load_scene_artifacts_restores_intrinsic_buffers_and_metadata(tmp_path):
    import json
    from relight_engine.workflows import load_scene_artifacts

    scene_dir = tmp_path / "intrinsic_scene"
    scene_dir.mkdir()
    h, w = 8, 10
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = -1.0
    albedo = np.full((h, w, 3), 0.55, np.float32)
    source_shading = np.full((h, w), 0.72, np.float32)
    residual = np.full((h, w, 3), 0.04, np.float32)
    np.savez_compressed(
        scene_dir / "scene_buffers.npz",
        rgb_srgb=np.full((h, w, 3), 0.4, np.float32),
        depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.full((h, w), 0.45, np.float32),
        geometry_confidence=np.full((h, w), 0.45, np.float32),
        normals=normals,
        lightable_mask=np.ones((h, w), np.float32),
        albedo=albedo,
        source_shading=source_shading,
        non_diffuse_residual=residual,
    )
    manifest = {
        "depth": {"depth_kind": "relative"},
        "geometry_confidence": {"source": "heuristic", "calibrated": False},
        "intrinsic_metadata": {"source": "test_intrinsic_estimator", "calibrated": False},
        "camera": {"fx": 10.0, "fy": 10.0, "cx": 5.0, "cy": 4.0},
    }
    (scene_dir / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    scene, _, _, _ = load_scene_artifacts(scene_dir)
    assert scene.albedo is not None and np.allclose(scene.albedo, albedo)
    assert scene.source_shading is not None and np.allclose(scene.source_shading, source_shading)
    assert scene.non_diffuse_residual is not None and np.allclose(scene.non_diffuse_residual, residual)
    assert scene.intrinsic_metadata["source"] == "test_intrinsic_estimator"


def test_run_scene_relight_supports_intrinsic_and_registered_relighter(tmp_path, monkeypatch):
    import json
    from relight_engine.adapters.protocols import NeuralRelightPrediction
    from relight_engine.adapters import registry
    from relight_engine.adapters.registry import register_relighter_backend
    from relight_engine.workflows import run_scene_relight

    monkeypatch.setattr(registry, "_RELIGHTER_BACKENDS", dict(registry._RELIGHTER_BACKENDS))

    class _TestRelighter:
        def render(self, sc, target_d, target_s, target_rig, settings):
            return NeuralRelightPrediction(
                rgb_srgb=np.full_like(sc.rgb_srgb, 0.7),
                metadata={"backend": "test_relighter"},
            )

        def runtime_snapshot(self):
            return {"device": "cpu"}

        def close(self):
            return {"released": False}

    register_relighter_backend("test_registered_relighter", lambda **_kw: _TestRelighter())

    scene_dir = tmp_path / "hybrid_scene"
    scene_dir.mkdir()
    h, w = 16, 20
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = -1.0
    np.savez_compressed(
        scene_dir / "scene_buffers.npz",
        rgb_srgb=np.full((h, w, 3), 0.35, np.float32),
        depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.ones((h, w), np.float32),
        geometry_confidence=np.ones((h, w), np.float32),
        normals=normals,
        lightable_mask=np.ones((h, w), np.float32),
        albedo=np.full((h, w, 3), 0.6, np.float32),
        source_shading=np.full((h, w), 0.25, np.float32),
    )
    manifest = {
        "depth": {"depth_kind": "relative"},
        "geometry_confidence": {"source": "heuristic", "calibrated": False},
        "camera": {"fx": 20.0, "fy": 20.0, "cx": 10.0, "cy": 8.0},
    }
    (scene_dir / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cfg = {
        "approved_for_test": True,
        "source_calibration": {"key_direction": [0.0, 0.0, -1.0]},
        "target_rig": {"lights": [{"kind": "directional", "direction": [0.0, 0.0, -1.0], "casts_shadows": False}]},
        "settings": {"base_blur_sigma": 3.0},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    rep_intr = run_scene_relight(scene_dir, cfg_path, tmp_path / "out_intr.png", renderer_mode="intrinsic")
    assert rep_intr["engine_metrics"]["renderer_mode"] == "intrinsic"

    rep_hyb = run_scene_relight(
        scene_dir, cfg_path, tmp_path / "out_hyb.png",
        renderer_mode="hybrid_neural", relighter_backend="test_registered_relighter",
    )
    assert rep_hyb["engine_metrics"]["renderer_mode"] == "hybrid_neural"
    assert rep_hyb["engine_metrics"]["neural_relighter"]["backend"] == "test_relighter"
