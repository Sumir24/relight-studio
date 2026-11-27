"""M1 Release Gate tests: Phase 1 contracts/serialization/caching + Phase 2 source-anchored relighting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from relight_engine import (
    CameraModel,
    EngineSettings,
    Light,
    LightingRig,
    SceneBuffers,
    prepare_relight_session,
    relight,
)
from relight_engine.calibration import assign_estimated_material_priors
from relight_engine.color import linear_to_srgb
from relight_engine.contracts import SCENE_SCHEMA_VERSION, validate_scene_schema_version
from relight_engine.lighting import resolve_effective_materials
from relight_engine.preview import downsample_scene_buffers_for_preview
from relight_engine.server.store import downsample_scene_for_preview
from relight_engine.workflows import load_scene_artifacts, serialize_scene_artifacts


def _full_scene(h: int = 24, w: int = 32) -> tuple[SceneBuffers, CameraModel]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.clip(0.35 + 0.10 * np.sin(xx * 0.5) * np.cos(yy * 0.5), 0.02, 0.96)[..., None]
    rgb = np.concatenate([rgb, np.clip(rgb * 0.85, 0.02, 0.96), np.clip(rgb * 1.10, 0.02, 0.96)], axis=-1).astype(np.float32)
    depth = np.where(xx < w // 2, 1.8, 3.0).astype(np.float32)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    protected = np.zeros((h, w), dtype=np.float32)
    protected[:4, :] = 1.0
    alpha = np.ones((h, w), dtype=np.float32)
    alpha[:, :3] = 0.0
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=depth,
        depth_confidence=np.full((h, w), 0.9, dtype=np.float32),
        lightable_mask=1.0 - protected,
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.full((h, w), 0.85, dtype=np.float32),
        alpha_mask=alpha,
        protected_mask=protected,
        semantic_masks={"foreground": (xx < w // 2).astype(np.float32)},
        instance_masks={"obj_1": ((xx >= 4) & (xx < 12)).astype(np.float32)},
        roughness=np.full((h, w), 0.32, dtype=np.float32),
        metalness=np.full((h, w), 0.78, dtype=np.float32),
        specular_level=np.full((h, w), 0.06, dtype=np.float32),
        specular_color=np.full((h, w, 3), np.array([0.85, 0.65, 0.45], np.float32), dtype=np.float32),
        clearcoat=np.full((h, w), 0.40, dtype=np.float32),
        clearcoat_roughness=np.full((h, w), 0.12, dtype=np.float32),
        subsurface=np.full((h, w), 0.22, dtype=np.float32),
        anisotropy=np.full((h, w), 0.65, dtype=np.float32),
        anisotropy_angle=np.full((h, w), 0.45, dtype=np.float32),
        material_confidence=np.full((h, w), 0.75, dtype=np.float32),
        material_metadata={"roughness": {"source": "model_estimated"}, "confidence_applied": True},
        albedo=linear_to_srgb(np.full((h, w, 3), 0.42, dtype=np.float32)),
        source_shading=np.full((h, w), 0.68, dtype=np.float32),
        non_diffuse_residual=np.full((h, w, 3), 0.09, dtype=np.float32),
        intrinsic_metadata={"backend": "test_intrinsic"},
        raw_material_maps={
            "roughness": np.full((h, w), 0.25, dtype=np.float32),
            "metalness": np.full((h, w), 0.88, dtype=np.float32),
        },
    )
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    return scene, cam


def test_phase1_canonical_scene_roundtrip_and_schema_compatibility(tmp_path):
    scene, cam = _full_scene()
    sdir = tmp_path / "scene_v2"
    serialize_scene_artifacts(sdir, scene, cam, manifest={"depth": {"model_id": "test_depth"}})
    loaded, loaded_cam, manifest, _ = load_scene_artifacts(sdir)

    assert loaded.schema_version == SCENE_SCHEMA_VERSION
    assert manifest["scene_schema_version"] == SCENE_SCHEMA_VERSION
    assert loaded_cam == cam

    for attr in (
        "rgb_srgb", "depth", "depth_confidence", "geometry_confidence", "normals",
        "lightable_mask", "protected_mask", "alpha_mask",
        "roughness", "metalness", "specular_level", "specular_color",
        "clearcoat", "clearcoat_roughness", "subsurface",
        "anisotropy", "anisotropy_angle", "material_confidence",
        "albedo", "source_shading", "non_diffuse_residual",
    ):
        a = getattr(scene, attr)
        b = getattr(loaded, attr)
        assert a is not None and b is not None, f"Missing round-trip field: {attr}"
        assert np.allclose(a, b, atol=1e-6), f"Mismatch in round-trip field: {attr}"

    assert sorted(loaded.raw_material_maps) == ["metalness", "roughness"]
    assert np.allclose(loaded.raw_material_maps["roughness"], scene.raw_material_maps["roughness"], atol=1e-6)
    assert loaded.intrinsic_metadata == scene.intrinsic_metadata
    assert loaded.material_metadata["confidence_applied"] is True

    # Incompatible future schema must fail explicitly
    bad_manifest = json.loads((sdir / "scene_manifest.json").read_text(encoding="utf-8"))
    bad_manifest["scene_schema_version"] = "99.0"
    (sdir / "scene_manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported scene schema version"):
        load_scene_artifacts(sdir)
    assert validate_scene_schema_version(None) == "1.0"


def test_phase1_shared_preview_complete_cache_keys_and_single_stage_confidence():
    scene, cam = _full_scene(48, 64)
    scene.clearcoat = np.zeros_like(scene.roughness)
    core_sc, core_cam = downsample_scene_buffers_for_preview(scene, cam, 32)
    srv_sc, srv_cam, srv_arr = downsample_scene_for_preview(scene, cam, {"clearcoat": scene.clearcoat}, 32)
    assert srv_cam == core_cam
    assert np.allclose(srv_sc.clearcoat, core_sc.clearcoat, atol=1e-6)
    assert np.allclose(srv_arr["clearcoat"], core_sc.clearcoat, atol=1e-6)

    # Preview cache must invalidate when secondary maps (e.g. clearcoat, subsurface, residual) change
    src_rig = LightingRig(ambient_strength=0.4, lights=(Light("s", "directional", direction=(0, 0, -1), strength=0.6, casts_shadows=False),))
    tgt_rig = LightingRig(ambient_strength=0.1, lights=(Light("k", "directional", direction=(-0.25, 0.1, -0.96), strength=2.0, casts_shadows=False),))
    cfg_prev = EngineSettings(renderer_mode="intrinsic", quality_mode="preview", preview_max_side=32, preview_upsample=True)

    r1 = relight(scene, cam, src_rig, tgt_rig, cfg_prev)
    scene.clearcoat = np.ones_like(scene.clearcoat)
    r2 = relight(scene, cam, src_rig, tgt_rig, cfg_prev)
    assert not np.allclose(r1.rgb_srgb, r2.rgb_srgb, atol=1e-4)

    # Single-stage confidence: assign_estimated_material_priors marks confidence_applied=True so lighting does NOT double-attenuate
    assigned = assign_estimated_material_priors(
        roughness=np.full((8, 8), 0.30, np.float32),
        metalness=np.full((8, 8), 0.92, np.float32),
        material_confidence=np.full((8, 8), 0.60, np.float32),
    )
    assert assigned.metadata["confidence_applied"] is True
    assert "roughness" in assigned.raw_maps
    mat_scene = SceneBuffers(
        rgb_srgb=np.full((8, 8, 3), 0.5, np.float32),
        depth=np.full((8, 8), 2.0, np.float32),
        depth_confidence=np.ones((8, 8), np.float32),
        lightable_mask=np.ones((8, 8), np.float32),
        depth_kind="relative",
        roughness=assigned.roughness,
        metalness=assigned.metalness,
        material_confidence=assigned.confidence,
        material_metadata=assigned.metadata,
        raw_material_maps=assigned.raw_maps,
    )
    eff = resolve_effective_materials(mat_scene)
    assert np.allclose(eff["roughness"], assigned.roughness, atol=1e-6)
    assert np.allclose(eff["metalness"], assigned.metalness, atol=1e-6)


def test_phase1_and_2_immutability_protection_noop_order_invariance_and_session_guards():
    scene, cam = _full_scene(28, 36)
    orig_rgb = scene.rgb_srgb.copy()
    orig_depth = scene.depth.copy()

    src_rig = LightingRig(ambient_strength=0.45, lights=(Light("src", "directional", direction=(0, 0, -1), strength=0.6, casts_shadows=False),))
    # Even biased albedo + residual must return EXACT original on identical source/target (no-op contract)
    scene.albedo = np.clip(scene.albedo * 1.6, 0.0, 1.0)
    noop = relight(scene, cam, src_rig, src_rig, EngineSettings(renderer_mode="intrinsic", refine_depth_edges=True))
    assert np.allclose(noop.rgb_srgb, orig_rgb, atol=1e-5)

    # Original arrays never mutated
    assert np.array_equal(scene.rgb_srgb, orig_rgb)
    assert np.array_equal(scene.depth, orig_depth)

    # RelightSession rejects hybrid_neural explicitly
    with pytest.raises(ValueError, match="hybrid_neural"):
        prepare_relight_session(scene, cam, src_rig, EngineSettings(renderer_mode="hybrid_neural"))
    sess = prepare_relight_session(scene, cam, src_rig, EngineSettings(renderer_mode="intrinsic"))
    with pytest.raises(ValueError, match="hybrid_neural"):
        sess.render(src_rig, EngineSettings(renderer_mode="hybrid_neural"))

    # Light-order invariance & strict protection of protected/alpha-zero pixels
    l1 = Light("l1", "directional", direction=(-0.7, 0.2, -0.68), color=(1.0, 0.3, 0.2), strength=1.3, casts_shadows=False)
    l2 = Light("l2", "directional", direction=(0.6, -0.1, -0.79), color=(0.2, 0.5, 1.0), strength=0.9, casts_shadows=False)
    rig_12 = LightingRig(ambient_strength=0.12, lights=(l1, l2))
    rig_21 = LightingRig(ambient_strength=0.12, lights=(l2, l1))

    res_12 = relight(scene, cam, src_rig, rig_12, EngineSettings(renderer_mode="intrinsic"))
    res_21 = relight(scene, cam, src_rig, rig_21, EngineSettings(renderer_mode="intrinsic"))
    assert np.allclose(res_12.rgb_srgb, res_21.rgb_srgb, atol=1e-5)

    # Fully protected rows (:4) and alpha==0 cols (:3) remain equal to normalized original
    assert np.allclose(res_12.rgb_srgb[:4, :], orig_rgb[:4, :], atol=1e-6)
    assert np.allclose(res_12.rgb_srgb[:, :3], orig_rgb[:, :3], atol=1e-6)
