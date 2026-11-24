from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from relight_engine.calibration import (
    MaterialAssignment, SourceLightCalibration, build_material_maps, kelvin_to_linear_rgb,
)


def test_source_light_calibration_builds_explicit_source_rig():
    calibration = SourceLightCalibration(
        key_direction=(0.0, 1.0, -1.0), key_strength=0.8,
        key_temperature_kelvin=4800, ambient_strength=0.12,
    )
    rig = calibration.to_rig()
    assert len(rig.lights) == 1
    assert rig.lights[0].kind == "directional"
    assert np.isclose(np.linalg.norm(rig.lights[0].direction), 1.0)
    assert calibration.to_metadata()["source"] == "user_calibration_not_automatic_estimation"


def test_kelvin_colour_is_linear_rgb_and_normalized():
    rgb = kelvin_to_linear_rgb(5600)
    assert len(rgb) == 3
    assert max(rgb) == pytest.approx(1.0)
    assert all(0 <= x <= 1 for x in rgb)


def test_invalid_source_light_direction_is_rejected():
    calibration = SourceLightCalibration(key_direction=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        calibration.to_rig()


def test_material_assignments_override_only_the_masked_region():
    mask = np.zeros((8, 10), np.float32)
    mask[2:6, 3:7] = 1.0
    assignment = MaterialAssignment("matte_foliage", mask, roughness=0.92, metalness=0.0, specular_level=0.035)
    maps = build_material_maps((8, 10), [assignment])
    assert np.allclose(maps.roughness[2:6, 3:7], 0.92)
    assert np.allclose(maps.roughness[0, :], 0.75)
    assert np.allclose(maps.coverage[2:6, 3:7], 1.0)
    assert maps.metadata["neural_material_estimation_used"] is False


def test_overlapping_material_masks_are_normalized_without_overshoot():
    a = np.full((4, 4), 0.8, np.float32)
    b = np.full((4, 4), 0.8, np.float32)
    maps = build_material_maps((4, 4), [
        MaterialAssignment("rough", a, roughness=1.0),
        MaterialAssignment("smooth", b, roughness=0.0),
    ])
    assert np.all(np.isfinite(maps.roughness))
    assert np.all((maps.roughness >= 0) & (maps.roughness <= 1))
    assert np.allclose(maps.roughness, 0.5)


def test_material_assignment_rejects_misaligned_mask():
    with pytest.raises(ValueError):
        build_material_maps((10, 10), [MaterialAssignment("bad", np.ones((4, 4), np.float32), 0.8)])



def test_material_assignment_builds_rgb_conductor_f0_and_separate_confidence():
    mask = np.zeros((8, 10), np.float32)
    mask[2:6, 3:7] = 1.0
    assignment = MaterialAssignment(
        "copper_trim",
        mask,
        roughness=0.24,
        metalness=1.0,
        specular_level=0.04,
        specular_color=(0.92, 0.52, 0.28),
        confidence=0.65,
    )
    maps = build_material_maps((8, 10), [assignment])

    assert maps.specular_color is not None
    assert maps.specular_color.shape == (8, 10, 3)
    assert np.allclose(maps.specular_color[3, 4], (0.92, 0.52, 0.28))
    assert np.allclose(maps.specular_color[0, 0], (0.65, 0.65, 0.65))
    assert maps.confidence[3, 4] == pytest.approx(0.65)
    assert maps.confidence[0, 0] == pytest.approx(0.0)
    assert maps.metadata["neural_material_estimation_used"] is False
    assert maps.metadata["assignments"][0]["confidence"] == pytest.approx(0.65)


def test_material_assignment_confidence_is_distinct_from_mask_blend():
    mask = np.full((4, 4), 0.75, np.float32)
    assignment = MaterialAssignment(
        "semi_gloss_panel",
        mask,
        roughness=0.35,
        blend=0.5,
        confidence=0.9,
    )
    maps = build_material_maps((4, 4), [assignment])

    assert np.allclose(maps.coverage, 0.375)
    assert np.allclose(maps.confidence, 0.9)
    assert np.allclose(maps.roughness, 0.75 + 0.375 * (0.35 - 0.75))


def test_material_assignment_rejects_invalid_rgb_f0_and_confidence():
    mask = np.ones((3, 3), np.float32)
    with pytest.raises(ValueError, match="specular_color"):
        build_material_maps((3, 3), [
            MaterialAssignment("bad_f0", mask, roughness=0.5, specular_color=(1.2, 0.0, 0.0))
        ])
    with pytest.raises(ValueError, match="confidence"):
        build_material_maps((3, 3), [
            MaterialAssignment("bad_confidence", mask, roughness=0.5, confidence=1.5)
        ])
