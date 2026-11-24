from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
import numpy as np

DepthKind = Literal['metric', 'relative']
LightKind = Literal['directional', 'point', 'area']


def _finite_scalar(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite.')


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera. Camera coordinates: +X right, +Y up, +Z away from camera."""
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError('Camera intrinsics must be finite.')
        if not (self.fx > 0 and self.fy > 0):
            raise ValueError('Camera focal lengths fx and fy must be positive.')


@dataclass
class SceneBuffers:
    """Shared image/scene data consumed by every lighting pass.

    rgb_srgb: HxWx3 float32 in [0,1].
    depth: HxW positive camera-Z depth, metric metres or relative scene units.
    depth_confidence, lightable_mask: HxW float32 in [0,1].
    Optional material maps are HxW float32 in [0,1]. Supplied normals are camera-space.
    """
    rgb_srgb: np.ndarray
    depth: np.ndarray
    depth_confidence: np.ndarray
    lightable_mask: np.ndarray
    depth_kind: DepthKind
    roughness: Optional[np.ndarray] = None
    metalness: Optional[np.ndarray] = None
    specular_level: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    geometry_confidence: Optional[np.ndarray] = None
    alpha_mask: Optional[np.ndarray] = None  # optional original transparency, preserved at output
    metadata: dict = field(default_factory=dict)
    protected_mask: Optional[np.ndarray] = None
    semantic_masks: dict[str, np.ndarray] = field(default_factory=dict)
    instance_masks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth.shape

    def validate(self) -> None:
        if self.depth_kind not in ('metric', 'relative'):
            raise ValueError("depth_kind must be 'metric' or 'relative'.")
        if self.rgb_srgb.ndim != 3 or self.rgb_srgb.shape[2] != 3:
            raise ValueError('rgb_srgb must have shape HxWx3.')
        h, w = self.rgb_srgb.shape[:2]
        if h < 2 or w < 2:
            raise ValueError('Scene buffers must be at least 2x2 pixels.')
        for name, arr in [('depth', self.depth), ('depth_confidence', self.depth_confidence),
                          ('lightable_mask', self.lightable_mask)]:
            if arr.shape != (h, w):
                raise ValueError(f'{name} must have shape {(h, w)}, got {arr.shape}.')
            if not np.isfinite(arr).all():
                raise ValueError(f'{name} must contain only finite values.')
        if not np.isfinite(self.rgb_srgb).all():
            raise ValueError('rgb_srgb must contain only finite values.')
        if np.any(self.rgb_srgb < 0) or np.any(self.rgb_srgb > 1):
            raise ValueError('rgb_srgb values must be in [0,1].')
        if np.any(self.depth < 0):
            raise ValueError('Depth must be non-negative; zero denotes invalid depth.')
        for name, arr in [('depth_confidence', self.depth_confidence), ('lightable_mask', self.lightable_mask),
                          ('roughness', self.roughness), ('metalness', self.metalness),
                          ('specular_level', self.specular_level), ('geometry_confidence', self.geometry_confidence),
                          ('alpha_mask', self.alpha_mask), ('protected_mask', self.protected_mask)]:
            if arr is not None:
                if arr.shape != (h, w):
                    raise ValueError(f'{name} must have shape {(h, w)}.')
                if not np.isfinite(arr).all():
                    raise ValueError(f'{name} must contain only finite values.')
                if np.any(arr < 0) or np.any(arr > 1):
                    raise ValueError(f'{name} values must be in [0,1].')
        if self.normals is not None:
            if self.normals.shape != (h, w, 3):
                raise ValueError('normals must have shape HxWx3.')
            if not np.isfinite(self.normals).all():
                raise ValueError('normals must contain only finite values.')
            norm = np.linalg.norm(self.normals, axis=-1)
            valid = self.depth > 0
            if np.any(valid & (norm < 1e-6)):
                raise ValueError('Valid-depth pixels must have non-zero supplied normals.')
        for family_name, masks in (("semantic", self.semantic_masks), ("instance", self.instance_masks)):
            for label, raw_mask in masks.items():
                mask = np.asarray(raw_mask)
                if mask.shape != (h, w):
                    raise ValueError(f'{family_name} mask {label!r} must have shape {(h, w)}.')
                if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
                    raise ValueError(f'{family_name} mask {label!r} values must be finite and in [0,1].')
        if np.any(self.depth_confidence < 0) or np.any(self.depth_confidence > 1):
            raise ValueError('depth_confidence must be in [0,1].')
        if np.any(self.lightable_mask < 0) or np.any(self.lightable_mask > 1):
            raise ValueError('lightable_mask must be in [0,1].')


@dataclass(frozen=True)
class Light:
    name: str
    kind: LightKind
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # linear RGB
    strength: float = 1.0
    direction: Optional[Tuple[float, float, float]] = None  # directional: from surface toward light
    position: Optional[Tuple[float, float, float]] = None   # point/area in camera coordinates
    area_size: Tuple[float, float] = (0.5, 0.5)             # area width/height, scene units
    samples: int = 4
    casts_shadows: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError('Light name cannot be empty.')
        if self.kind not in ('directional', 'point', 'area'):
            raise ValueError(f'Unsupported light kind: {self.kind}')
        _finite_scalar('Light strength', self.strength)
        if self.strength < 0:
            raise ValueError('Light strength cannot be negative.')
        color = np.asarray(self.color, dtype=np.float64)
        if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0):
            raise ValueError('Light color must be a finite non-negative linear RGB triple.')
        if self.kind == 'directional':
            if self.direction is None:
                raise ValueError('Directional light requires direction.')
            direction = np.asarray(self.direction, dtype=np.float64)
            if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
                raise ValueError('Directional light direction must be a finite non-zero 3-vector.')
        if self.kind in ('point', 'area'):
            if self.position is None:
                raise ValueError(f'{self.kind} light requires position.')
            position = np.asarray(self.position, dtype=np.float64)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('Light position must be a finite 3-vector.')
        area_size = np.asarray(self.area_size, dtype=np.float64)
        if area_size.shape != (2,) or not np.isfinite(area_size).all() or np.any(area_size < 0):
            raise ValueError('Area-light dimensions must be a finite non-negative pair.')
        if self.kind == 'area' and (not isinstance(self.samples, int) or self.samples < 1):
            raise ValueError('Area light samples must be an integer >= 1.')


@dataclass(frozen=True)
class LightingRig:
    ambient_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient_strength: float = 0.15
    lights: Tuple[Light, ...] = ()

    def validate(self) -> None:
        _finite_scalar('Ambient strength', self.ambient_strength)
        if self.ambient_strength < 0:
            raise ValueError('Ambient strength cannot be negative.')
        color = np.asarray(self.ambient_color, dtype=np.float64)
        if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0):
            raise ValueError('Ambient color must be a finite non-negative RGB triple.')
        for light in self.lights:
            light.validate()


@dataclass(frozen=True)
class EngineSettings:
    base_blur_sigma: float = 18.0
    detail_power: float = 0.78
    ratio_min: float = 0.25
    ratio_max: float = 4.0
    color_ratio_min: float = 0.55
    color_ratio_max: float = 1.8
    specular_scale: float = 1.0
    shadow_steps: int = 12
    shadow_bias_metric: float = 0.015
    shadow_bias_relative: float = 0.003
    min_geometry_confidence: float = 0.15
    output_exposure: float = 1.0

    def validate(self) -> None:
        numeric = {
            'base_blur_sigma': self.base_blur_sigma,
            'detail_power': self.detail_power,
            'ratio_min': self.ratio_min,
            'ratio_max': self.ratio_max,
            'color_ratio_min': self.color_ratio_min,
            'color_ratio_max': self.color_ratio_max,
            'specular_scale': self.specular_scale,
            'shadow_bias_metric': self.shadow_bias_metric,
            'shadow_bias_relative': self.shadow_bias_relative,
            'min_geometry_confidence': self.min_geometry_confidence,
            'output_exposure': self.output_exposure,
        }
        for name, value in numeric.items():
            _finite_scalar(name, value)
        if not (self.base_blur_sigma > 0 and self.ratio_min > 0 and self.ratio_max >= self.ratio_min):
            raise ValueError('Invalid blur sigma or shading-ratio limits.')
        if not (self.color_ratio_min > 0 and self.color_ratio_max >= self.color_ratio_min):
            raise ValueError('Invalid colour-ratio limits.')
        if self.detail_power < 0 or self.specular_scale < 0 or self.output_exposure < 0:
            raise ValueError('Detail, specular and exposure controls must be non-negative.')
        if self.shadow_bias_metric < 0 or self.shadow_bias_relative < 0:
            raise ValueError('Shadow bias values must be non-negative.')
        if not isinstance(self.shadow_steps, int) or self.shadow_steps < 2:
            raise ValueError('shadow_steps must be an integer >= 2.')
        if not 0 <= self.min_geometry_confidence <= 1:
            raise ValueError('min_geometry_confidence must be in [0,1].')


@dataclass
class RelightResult:
    rgb_srgb: np.ndarray
    normal_map: np.ndarray
    diffuse_ratio: np.ndarray
    shadow_visibility: np.ndarray
    geometry_confidence: np.ndarray
    metrics: dict
