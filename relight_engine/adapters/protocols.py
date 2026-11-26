"""Model-agnostic adapter protocols and neutral prediction containers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
import numpy as np

from ..contracts import EngineSettings, LightingRig, SceneBuffers
from ..scene_policy import SemanticRoutingPolicy


@dataclass
class NormalPrediction:
    """Camera-space surface normals (+X right, +Y up, +Z away from camera)."""
    normals: np.ndarray
    confidence: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class IntrinsicPrediction:
    """Intrinsic image decomposition aligned to the source pixel grid."""
    albedo: np.ndarray
    source_shading: np.ndarray
    non_diffuse_residual: np.ndarray | None = None
    roughness: np.ndarray | None = None
    metalness: np.ndarray | None = None
    specular_level: np.ndarray | None = None
    specular_color: np.ndarray | None = None
    clearcoat: np.ndarray | None = None
    clearcoat_roughness: np.ndarray | None = None
    anisotropy: np.ndarray | None = None
    material_confidence: np.ndarray | None = None
    roughness_uncertainty: np.ndarray | None = None
    metalness_uncertainty: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class NeuralRelightPrediction:
    """Output returned by a pluggable neural relighting renderer."""
    rgb_srgb: np.ndarray
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class DepthEstimator(Protocol):
    def estimate(self, rgb_srgb: np.ndarray) -> Any: ...
    def runtime_snapshot(self) -> dict: ...
    def close(self) -> dict: ...


@runtime_checkable
class NormalEstimator(Protocol):
    def estimate(
        self, rgb_srgb: np.ndarray, *, depth: np.ndarray | None = None
    ) -> NormalPrediction: ...
    def runtime_snapshot(self) -> dict: ...
    def close(self) -> dict: ...


@runtime_checkable
class IntrinsicEstimator(Protocol):
    def estimate(self, rgb_srgb: np.ndarray) -> IntrinsicPrediction: ...
    def runtime_snapshot(self) -> dict: ...
    def close(self) -> dict: ...


@runtime_checkable
class NeuralRelighter(Protocol):
    def render(
        self,
        scene: SceneBuffers,
        target_diffuse: np.ndarray,
        target_specular: np.ndarray,
        target_rig: LightingRig,
        settings: EngineSettings,
    ) -> NeuralRelightPrediction: ...
    def runtime_snapshot(self) -> dict: ...
    def close(self) -> dict: ...


@runtime_checkable
class SceneDiscoverer(Protocol):
    def discover(self, rgb_srgb: np.ndarray) -> Any: ...
    def runtime_snapshot(self) -> dict: ...
    def close(self) -> dict: ...


@runtime_checkable
class SceneSegmenter(Protocol):
    def estimate(
        self,
        rgb_srgb: np.ndarray,
        prompts: tuple[str, ...] | list[str],
        *,
        policy: SemanticRoutingPolicy | None = None,
    ) -> Any: ...
    def runtime_snapshot(self) -> dict: ...
    def close(self) -> dict: ...
