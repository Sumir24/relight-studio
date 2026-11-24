from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from relight_engine.adapters import (
    DepthPrediction, SemanticPrediction, build_scene_buffers,
    camera_from_horizontal_fov, compose_scene_masks, normalize_relative_depth, SemanticRoutingPolicy,
)


def test_semantic_policy_is_domain_agnostic_and_has_no_hardcoded_protected_classes():
    shape = (10, 12)
    person = np.zeros(shape, np.float32); person[1:8, 2:6] = 0.9
    glass_bottle = np.zeros(shape, np.float32); glass_bottle[3:9, 7:10] = 0.8
    lightable, protected = compose_scene_masks({"person": person, "glass bottle": glass_bottle}, shape=shape)
    assert np.all(lightable == 1.0)
    assert np.all(protected == 0.0)


def test_semantic_policy_can_protect_arbitrary_labels():
    shape = (10, 12)
    face = np.zeros(shape, np.float32); face[:4, :5] = 0.95
    shirt = np.zeros(shape, np.float32); shirt[4:9, :5] = 0.85
    lightable, protected = compose_scene_masks(
        {"face": face, "shirt": shirt}, shape=shape,
        policy=SemanticRoutingPolicy(protected_labels=("face",)),
    )
    assert np.all(protected[:4, :5] > 0.9)
    # Soft masks retain feathering at uncertain boundaries; high-confidence face core is suppressed.
    assert np.all(lightable[:4, :5] <= 0.051)
    assert np.all(lightable[5:8, :5] == 1.0)


def test_semantic_policy_can_restrict_lightable_labels_without_taxonomy():
    shape = (8, 8)
    mug = np.zeros(shape, np.float32); mug[1:4, 1:4] = 0.7
    table = np.zeros(shape, np.float32); table[4:7, 4:7] = 0.8
    lightable, protected = compose_scene_masks(
        {"ceramic mug": mug, "wooden table": table}, shape=shape,
        policy=SemanticRoutingPolicy(lightable_labels=("ceramic mug",)),
    )
    assert lightable[2, 2] == pytest.approx(0.7)
    assert lightable[5, 5] == 0.0
    assert np.all(protected == 0.0)


def test_semantic_policy_rejects_misaligned_masks():
    with pytest.raises(ValueError):
        compose_scene_masks({"sofa": np.zeros((10, 10), np.float32),
                             "person": np.zeros((9, 10), np.float32)})


def test_semantic_policy_can_run_without_any_detected_objects():
    lightable, protected = compose_scene_masks({}, shape=(5, 7))
    assert np.all(lightable == 1.0)
    assert np.all(protected == 0.0)


def test_semantic_policy_rejects_colliding_normalized_labels():
    with pytest.raises(ValueError):
        compose_scene_masks({"red car": np.zeros((4, 4), np.float32),
                             "red-car": np.zeros((4, 4), np.float32)})


def test_scene_domain_hint_is_advisory_and_accepts_general_domains():
    from relight_engine.scene_domains import SceneDomainHint
    hint = SceneDomainHint(domain="product", confidence=0.82, source="test_fixture")
    hint.validate()
    assert hint.to_metadata()["affects_renderer"] is False



def test_same_semantic_policy_accepts_all_target_photo_domains():
    from relight_engine.scene_domains import SUPPORTED_QA_DOMAINS
    shape = (16, 16)
    for index, domain in enumerate(SUPPORTED_QA_DOMAINS):
        label = f"{domain}_subject_{index}"
        mask = np.zeros(shape, np.float32)
        mask[2:12, 3:13] = 0.75
        lightable, protected = compose_scene_masks({label: mask}, shape=shape)
        assert lightable[5, 5] == pytest.approx(1.0)
        assert protected[5, 5] == pytest.approx(0.0)


def test_fov_camera_is_explicitly_approximate_by_default():
    result = camera_from_horizontal_fov(1536, 1152, 60.0, source="assumed")
    assert result.camera.fx > 0 and result.camera.fy > 0
    assert result.calibrated is False
    assert "approximation" in result.note


def test_calibrated_camera_source_is_marked():
    result = camera_from_horizontal_fov(1536, 1152, 60.0, source="user_calibrated")
    assert result.calibrated is True


def test_scene_factory_preserves_masks_and_no_material_fabrication():
    h, w = 8, 9
    rgb = np.full((h, w, 3), 0.25, np.float32)
    depth = DepthPrediction(np.full((h, w), 3.0, np.float32), np.full((h, w), 0.4, np.float32), "relative")
    lightable = np.zeros((h, w), np.float32); lightable[2:6, 2:6] = 0.8
    protected = np.zeros((h, w), np.float32); protected[:2] = 1
    sem = SemanticPrediction(lightable_mask=lightable, protected_mask=protected,
                             class_masks={"portrait subject": protected, "jacket": lightable},
                             metadata={"material_maps_estimated": False})
    scene = build_scene_buffers(rgb, depth, sem)
    scene.validate()
    assert scene.roughness is None
    assert scene.metalness is None
    assert scene.metadata["material_maps_estimated"] is False
    assert scene.metadata["semantic_routing_is_domain_agnostic"] is True
    assert scene.metadata["protected_region_fraction"] > 0


def test_scene_factory_keeps_optional_alpha_mask():
    h, w = 7, 9
    rgb = np.full((h, w, 3), 0.3, np.float32)
    depth = DepthPrediction(np.ones((h, w), np.float32), np.ones((h, w), np.float32), "relative")
    semantics = SemanticPrediction(lightable_mask=np.ones((h, w), np.float32))
    alpha = np.ones((h, w), np.float32); alpha[:, :3] = 0.0
    scene = build_scene_buffers(rgb, depth, semantics, alpha_mask=alpha)
    scene.validate()
    assert np.array_equal(scene.alpha_mask, alpha)
    assert scene.metadata["alpha_preserved"] is True


def test_relative_depth_polarity_maps_near_to_smaller_camera_z():
    raw = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    near_high = normalize_relative_depth(raw, near_is_high=True)
    far_high = normalize_relative_depth(raw, near_is_high=False)
    assert near_high[0, 0] > near_high[-1, -1]
    assert far_high[0, 0] < far_high[-1, -1]
    assert np.isclose(near_high.min(), 1.0)
    assert np.isclose(near_high.max(), 5.0)


def test_relative_depth_rejects_flat_or_nonfinite_predictions():
    with pytest.raises(ValueError):
        normalize_relative_depth(np.ones((5, 5), np.float32), near_is_high=True)
    raw = np.ones((5, 5), np.float32); raw[0, 0] = np.nan
    with pytest.raises(ValueError):
        normalize_relative_depth(raw, near_is_high=True)


def test_depth_polarity_inspection_returns_inverse_candidate_maps():
    from relight_engine.adapters import relative_depth_polarity_candidates
    raw = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    candidates = relative_depth_polarity_candidates(raw)
    near_high = candidates["near_is_high"]
    far_high = candidates["far_is_high"]
    assert np.allclose(near_high + far_high, 6.0)
    assert near_high.shape == raw.shape
    assert float(near_high[0, 0]) > float(near_high[-1, -1])
    assert float(far_high[0, 0]) < float(far_high[-1, -1])


def test_scene_factory_preserves_arbitrary_per_instance_mask_keys():
    h, w = 9, 11
    rgb = np.full((h, w, 3), 0.2, np.float32)
    depth = DepthPrediction(np.full((h, w), 2.0, np.float32), np.ones((h, w), np.float32), "relative")
    first_person = np.zeros((h, w), np.float32); first_person[1:5, 1:4] = 0.9
    second_person = np.zeros((h, w), np.float32); second_person[3:8, 7:10] = 0.85
    semantics = SemanticPrediction(
        class_masks={"person": np.maximum(first_person, second_person)},
        instance_masks={"person__001": first_person, "person__002": second_person},
    )
    scene = build_scene_buffers(rgb, depth, semantics)
    scene.validate()
    assert scene.metadata["semantic_label_keys"] == ["person"]
    assert scene.metadata["instance_mask_keys"] == ["person__001", "person__002"]


def test_scene_factory_rejects_misaligned_or_invalid_instance_masks():
    h, w = 6, 7
    rgb = np.full((h, w, 3), 0.2, np.float32)
    depth = DepthPrediction(np.ones((h, w), np.float32), np.ones((h, w), np.float32), "relative")
    with pytest.raises(ValueError, match="exactly match"):
        build_scene_buffers(rgb, depth, SemanticPrediction(instance_masks={"object__001": np.zeros((h, w - 1), np.float32)}))
    invalid = np.zeros((h, w), np.float32); invalid[0, 0] = 1.5
    with pytest.raises(ValueError, match=r"finite and in \[0,1\]"):
        build_scene_buffers(rgb, depth, SemanticPrediction(instance_masks={"object__001": invalid}))
