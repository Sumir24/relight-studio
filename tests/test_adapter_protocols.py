from __future__ import annotations

import numpy as np
import pytest

from relight_engine.adapters.protocols import (
    DepthEstimator,
    IntrinsicEstimator,
    IntrinsicPrediction,
    NeuralRelightPrediction,
    NeuralRelighter,
    NormalEstimator,
    NormalPrediction,
    SceneDiscoverer,
    SceneSegmenter,
)
from relight_engine.adapters.registry import (
    available_backends,
    create_intrinsic_backend,
    create_normal_backend,
    create_relighter_backend,
    register_intrinsic_backend,
    register_normal_backend,
    register_relighter_backend,
)
from relight_engine.contracts import EngineSettings, LightingRig, SceneBuffers


class _DummyNormals:
    def estimate(self, rgb_srgb: np.ndarray, *, depth: np.ndarray | None = None) -> NormalPrediction:
        h, w = rgb_srgb.shape[:2]
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[..., 2] = -1.0
        return NormalPrediction(normals=normals, confidence=np.ones((h, w), dtype=np.float32))

    def runtime_snapshot(self) -> dict:
        return {"device": "cpu", "model_loaded": False}

    def close(self) -> dict:
        return {"released": False}


class _DummyIntrinsics:
    def estimate(self, rgb_srgb: np.ndarray) -> IntrinsicPrediction:
        h, w = rgb_srgb.shape[:2]
        return IntrinsicPrediction(
            albedo=np.full((h, w, 3), 0.5, dtype=np.float32),
            source_shading=np.ones((h, w), dtype=np.float32),
        )

    def runtime_snapshot(self) -> dict:
        return {"device": "cpu", "model_loaded": False}

    def close(self) -> dict:
        return {"released": False}


class _DummyRelighter:
    def render(
        self,
        scene: SceneBuffers,
        target_diffuse: np.ndarray,
        target_specular: np.ndarray,
        target_rig: LightingRig,
        settings: EngineSettings,
    ) -> NeuralRelightPrediction:
        return NeuralRelightPrediction(rgb_srgb=scene.rgb_srgb.copy())

    def runtime_snapshot(self) -> dict:
        return {"device": "cpu", "model_loaded": False}

    def close(self) -> dict:
        return {"released": False}


def test_custom_in_memory_adapters_satisfy_protocols_and_registry(monkeypatch):
    from relight_engine.adapters import registry

    monkeypatch.setattr(registry, "_NORMAL_BACKENDS", dict(registry._NORMAL_BACKENDS))
    monkeypatch.setattr(registry, "_INTRINSIC_BACKENDS", dict(registry._INTRINSIC_BACKENDS))
    monkeypatch.setattr(registry, "_RELIGHTER_BACKENDS", dict(registry._RELIGHTER_BACKENDS))

    normals = _DummyNormals()
    intrinsics = _DummyIntrinsics()
    relighter = _DummyRelighter()

    assert isinstance(normals, NormalEstimator)
    assert isinstance(intrinsics, IntrinsicEstimator)
    assert isinstance(relighter, NeuralRelighter)

    register_normal_backend("test_dummy_normals", lambda **_kw: _DummyNormals())
    register_intrinsic_backend("test_dummy_intrinsics", lambda **_kw: _DummyIntrinsics())
    register_relighter_backend("test_dummy_relighter", lambda **_kw: _DummyRelighter())

    assert "test_dummy_normals" in available_backends()["normals"]
    assert "test_dummy_intrinsics" in available_backends()["intrinsics"]
    assert "test_dummy_relighter" in available_backends()["relighters"]

    assert isinstance(create_normal_backend("test_dummy_normals"), NormalEstimator)
    assert isinstance(create_intrinsic_backend("test_dummy_intrinsics"), IntrinsicEstimator)
    assert isinstance(create_relighter_backend("test_dummy_relighter"), NeuralRelighter)

    with pytest.raises(ValueError, match="Unknown"):
        create_normal_backend("nonexistent_backend")
    with pytest.raises(ValueError, match="Unknown"):
        create_intrinsic_backend("nonexistent_backend")
    with pytest.raises(ValueError, match="Unknown"):
        create_relighter_backend("nonexistent_backend")


def test_existing_adapters_conform_to_protocols():
    from relight_engine.adapters import (
        Florence2SceneDiscovery,
        HuggingFaceDepthAnythingV2,
        HuggingFaceGroundedSAM2,
    )

    assert isinstance(HuggingFaceDepthAnythingV2(relative_near_is_high=True), DepthEstimator)
    assert isinstance(Florence2SceneDiscovery(), SceneDiscoverer)
    assert isinstance(HuggingFaceGroundedSAM2(), SceneSegmenter)
