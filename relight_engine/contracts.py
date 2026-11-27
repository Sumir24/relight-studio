from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
import math
import numpy as np

DepthKind = Literal['metric', 'relative']
LightKind = Literal['directional', 'point', 'area', 'spot']
RendererMode = Literal['auto', 'ratio', 'intrinsic', 'hybrid_neural']
DetailPreservationMode = Literal['baseline', 'f3', 'wavelet']
RenderQualityMode = Literal['standard', 'preview', 'final']
RelightOperation = Literal['replace', 'add']
IntensityMode = Literal['artistic', 'physical']

SCENE_SCHEMA_VERSION = '2.0'
SUPPORTED_SCENE_SCHEMA_VERSIONS = ('1.0', '1.1', '2.0')


def validate_scene_schema_version(version: object) -> str:
    """Validate serialized scene schema compatibility (accepts 1.0, 1.1, and 2.0)."""
    if version is None:
        return '1.0'
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f'Invalid scene schema version: {version!r}.')
    ver = version.strip()
    if ver not in SUPPORTED_SCENE_SCHEMA_VERSIONS:
        raise ValueError(
            f'Unsupported scene schema version {ver!r}; supported versions: {SUPPORTED_SCENE_SCHEMA_VERSIONS}.'
        )
    return ver


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
    clearcoat: Optional[np.ndarray] = None
    clearcoat_roughness: Optional[np.ndarray] = None
    subsurface: Optional[np.ndarray] = None
    anisotropy: Optional[np.ndarray] = None
    anisotropy_angle: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    geometry_confidence: Optional[np.ndarray] = None
    alpha_mask: Optional[np.ndarray] = None  # optional original transparency, preserved at output
    metadata: dict = field(default_factory=dict)
    protected_mask: Optional[np.ndarray] = None
    semantic_masks: dict[str, np.ndarray] = field(default_factory=dict)
    instance_masks: dict[str, np.ndarray] = field(default_factory=dict)
    # RGB conductor F0 / specular reflectance in linear RGB, not an inferred value.
    specular_color: Optional[np.ndarray] = None
    # Reliability/authority for supplied material maps; not a calibrated probability by default.
    material_confidence: Optional[np.ndarray] = None
    # Per-map provenance, e.g. {"roughness": {"source": "user_authored"}}.
    material_metadata: dict = field(default_factory=dict)
    # Optional model-neutral intrinsic decomposition buffers (albedo in [0,1], shading/residual >= 0).
    albedo: Optional[np.ndarray] = None
    source_shading: Optional[np.ndarray] = None
    non_diffuse_residual: Optional[np.ndarray] = None
    intrinsic_metadata: dict = field(default_factory=dict)
    # Raw uncalibrated model material predictions (e.g. roughness, metalness) kept separate from processed priors.
    raw_material_maps: dict[str, np.ndarray] = field(default_factory=dict)
    schema_version: str = SCENE_SCHEMA_VERSION

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
                          ('specular_level', self.specular_level), ('clearcoat', self.clearcoat),
                          ('clearcoat_roughness', self.clearcoat_roughness), ('subsurface', self.subsurface),
                          ('anisotropy', self.anisotropy),
                          ('geometry_confidence', self.geometry_confidence),
                          ('alpha_mask', self.alpha_mask), ('protected_mask', self.protected_mask),
                          ('material_confidence', self.material_confidence)]:
            if arr is not None:
                if arr.shape != (h, w):
                    raise ValueError(f'{name} must have shape {(h, w)}.')
                if not np.isfinite(arr).all():
                    raise ValueError(f'{name} must contain only finite values.')
                if np.any(arr < 0) or np.any(arr > 1):
                    raise ValueError(f'{name} values must be in [0,1].')
        if self.anisotropy_angle is not None:
            if self.anisotropy_angle.shape != (h, w):
                raise ValueError(f'anisotropy_angle must have shape {(h, w)}.')
            if not np.isfinite(self.anisotropy_angle).all():
                raise ValueError('anisotropy_angle must contain only finite values.')
        if self.specular_color is not None:
            if self.specular_color.shape != (h, w, 3):
                raise ValueError('specular_color must have shape HxWx3.')
            if not np.isfinite(self.specular_color).all():
                raise ValueError('specular_color must contain only finite values.')
            if np.any(self.specular_color < 0) or np.any(self.specular_color > 1):
                raise ValueError('specular_color values must be in [0,1].')
        if not isinstance(self.material_metadata, dict):
            raise ValueError('material_metadata must be a dictionary.')
        if not isinstance(self.intrinsic_metadata, dict):
            raise ValueError('intrinsic_metadata must be a dictionary.')
        if self.albedo is not None:
            if self.albedo.shape != (h, w, 3):
                raise ValueError('albedo must have shape HxWx3.')
            if not np.isfinite(self.albedo).all() or np.any(self.albedo < 0) or np.any(self.albedo > 1):
                raise ValueError('albedo values must be finite and in [0,1].')
        if self.source_shading is not None:
            if self.source_shading.shape not in ((h, w), (h, w, 3)):
                raise ValueError('source_shading must have shape HxW or HxWx3.')
            if not np.isfinite(self.source_shading).all() or np.any(self.source_shading < 0):
                raise ValueError('source_shading values must be finite and non-negative.')
        if self.non_diffuse_residual is not None:
            if self.non_diffuse_residual.shape != (h, w, 3):
                raise ValueError('non_diffuse_residual must have shape HxWx3.')
            if not np.isfinite(self.non_diffuse_residual).all() or np.any(self.non_diffuse_residual < 0):
                raise ValueError('non_diffuse_residual values must be finite and non-negative.')
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
        validate_scene_schema_version(self.schema_version)
        if not isinstance(self.raw_material_maps, dict):
            raise ValueError('raw_material_maps must be a dictionary.')
        for raw_name, raw_arr in self.raw_material_maps.items():
            arr = np.asarray(raw_arr)
            if raw_name == 'specular_color':
                if arr.shape != (h, w, 3):
                    raise ValueError(f'raw_material_maps[{raw_name!r}] must have shape HxWx3.')
            elif arr.shape != (h, w):
                raise ValueError(f'raw_material_maps[{raw_name!r}] must have shape {(h, w)}.')
            if not np.isfinite(arr).all():
                raise ValueError(f'raw_material_maps[{raw_name!r}] must contain only finite values.')
            if raw_name != 'anisotropy_angle' and (np.any(arr < 0) or np.any(arr > 1)):
                raise ValueError(f'raw_material_maps[{raw_name!r}] values must be in [0,1].')


@dataclass(frozen=True)
class Light:
    name: str
    kind: LightKind
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # linear RGB
    strength: float = 1.0
    direction: Optional[Tuple[float, float, float]] = None  # directional: toward light; spot/area: emitter facing direction
    position: Optional[Tuple[float, float, float]] = None   # point/area/spot in camera coordinates
    area_size: Tuple[float, float] = (0.5, 0.5)             # area width/height, scene units
    samples: int = 4
    casts_shadows: bool = True
    wrap: float = 0.0                                       # wrapped Lambertian scatter [0, 1]
    radius: float = 0.0                                     # finite source radius for soft penumbras >= 0
    light_id: Optional[str] = None                          # stable identifier for caching and per-light records
    spot_inner_angle_deg: float = 20.0                      # spotlight full-intensity cone half-angle (degrees)
    spot_outer_angle_deg: float = 40.0                      # spotlight zero-intensity cutoff half-angle (degrees)
    intensity_mode: IntensityMode = 'artistic'              # 'artistic' or calibrated 'physical'

    @property
    def resolved_id(self) -> str:
        return self.light_id if (self.light_id and self.light_id.strip()) else self.name

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError('Light name cannot be empty.')
        if self.kind not in ('directional', 'point', 'area', 'spot'):
            raise ValueError(f'Unsupported light kind: {self.kind}')
        if self.intensity_mode not in ('artistic', 'physical'):
            raise ValueError("Light intensity_mode must be 'artistic' or 'physical'.")
        _finite_scalar('Light strength', self.strength)
        if self.strength < 0:
            raise ValueError('Light strength cannot be negative.')
        _finite_scalar('Light wrap', self.wrap)
        if not 0.0 <= self.wrap <= 1.0:
            raise ValueError('Light wrap must be in [0,1].')
        _finite_scalar('Light radius', self.radius)
        if self.radius < 0.0:
            raise ValueError('Light radius must be non-negative.')
        _finite_scalar('Light spot_inner_angle_deg', self.spot_inner_angle_deg)
        _finite_scalar('Light spot_outer_angle_deg', self.spot_outer_angle_deg)
        if not (0.0 < self.spot_inner_angle_deg <= self.spot_outer_angle_deg <= 180.0):
            raise ValueError('Spotlight angles must satisfy 0 < spot_inner_angle_deg <= spot_outer_angle_deg <= 180.')
        color = np.asarray(self.color, dtype=np.float64)
        if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0):
            raise ValueError('Light color must be a finite non-negative linear RGB triple.')
        if self.kind == 'directional':
            if self.direction is None:
                raise ValueError('Directional light requires direction.')
            direction = np.asarray(self.direction, dtype=np.float64)
            if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
                raise ValueError('Directional light direction must be a finite non-zero 3-vector.')
        if self.kind in ('point', 'area', 'spot'):
            if self.position is None:
                raise ValueError(f'{self.kind} light requires position.')
            position = np.asarray(self.position, dtype=np.float64)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('Light position must be a finite 3-vector.')
        if self.kind == 'spot':
            if self.direction is None:
                raise ValueError('Spotlight requires an emitter facing direction.')
            direction = np.asarray(self.direction, dtype=np.float64)
            if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
                raise ValueError('Spotlight direction must be a finite non-zero 3-vector.')
        if self.kind == 'area' and self.direction is not None:
            direction = np.asarray(self.direction, dtype=np.float64)
            if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
                raise ValueError('Area light direction must be a finite non-zero 3-vector when set.')
        area_size = np.asarray(self.area_size, dtype=np.float64)
        if area_size.shape != (2,) or not np.isfinite(area_size).all() or np.any(area_size < 0):
            raise ValueError('Area-light dimensions must be a finite non-negative pair.')
        if self.kind == 'area' and (not isinstance(self.samples, int) or self.samples < 1):
            raise ValueError('Area light samples must be an integer >= 1.')


@dataclass(frozen=True)
class EnvironmentLight:
    """Optional parameterized studio dome or equirectangular HDRI environment-light interface."""
    sky_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    horizon_color: Tuple[float, float, float] = (0.75, 0.78, 0.85)
    ground_color: Tuple[float, float, float] = (0.20, 0.20, 0.22)
    strength: float = 0.0
    key_azimuth_deg: float = 0.0
    key_elevation_deg: float = 35.0
    key_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    key_strength: float = 0.0
    key_sharpness: float = 12.0
    # Optional HxWx3 finite non-negative linear-RGB equirectangular (lat-long) HDRI map.
    # Convention: u=0.5 faces -Z (toward camera), u=0.75 faces +X, v=0 is zenith (+Y).
    env_map: Optional[np.ndarray] = field(default=None, repr=False, hash=False, compare=False)
    env_map_strength: float = 1.0
    diffuse_strength: float = 1.0
    specular_strength: float = 1.0
    rotation_deg: float = 0.0

    def validate(self) -> None:
        for name, val in (
            ('strength', self.strength),
            ('key_azimuth_deg', self.key_azimuth_deg),
            ('key_elevation_deg', self.key_elevation_deg),
            ('key_strength', self.key_strength),
            ('key_sharpness', self.key_sharpness),
            ('env_map_strength', self.env_map_strength),
            ('diffuse_strength', self.diffuse_strength),
            ('specular_strength', self.specular_strength),
            ('rotation_deg', self.rotation_deg),
        ):
            _finite_scalar(f'EnvironmentLight {name}', val)
        if (
            self.strength < 0.0
            or self.key_strength < 0.0
            or self.env_map_strength < 0.0
            or self.diffuse_strength < 0.0
            or self.specular_strength < 0.0
        ):
            raise ValueError('EnvironmentLight strengths must be non-negative.')
        if self.key_sharpness <= 0.0:
            raise ValueError('EnvironmentLight key_sharpness must be positive.')
        for cname, cval in (
            ('sky_color', self.sky_color),
            ('horizon_color', self.horizon_color),
            ('ground_color', self.ground_color),
            ('key_color', self.key_color),
        ):
            arr = np.asarray(cval, dtype=np.float64)
            if arr.shape != (3,) or not np.isfinite(arr).all() or np.any(arr < 0):
                raise ValueError(f'EnvironmentLight {cname} must be a finite non-negative RGB triple.')
        if self.env_map is not None:
            env = np.asarray(self.env_map, dtype=np.float32)
            if env.ndim != 3 or env.shape[2] != 3 or env.shape[0] < 2 or env.shape[1] < 4:
                raise ValueError('EnvironmentLight env_map must have shape HxWx3 with H>=2 and W>=4.')
            if not np.isfinite(env).all() or np.any(env < 0):
                raise ValueError('EnvironmentLight env_map must contain finite non-negative linear radiance values.')


@dataclass(frozen=True)
class LightingRig:
    ambient_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient_strength: float = 0.15
    lights: Tuple[Light, ...] = ()
    ambient_ground_color: Optional[Tuple[float, float, float]] = None
    environment: Optional[EnvironmentLight] = None

    def validate(self) -> None:
        _finite_scalar('Ambient strength', self.ambient_strength)
        if self.ambient_strength < 0:
            raise ValueError('Ambient strength cannot be negative.')
        color = np.asarray(self.ambient_color, dtype=np.float64)
        if color.shape != (3,) or not np.isfinite(color).all() or np.any(color < 0):
            raise ValueError('Ambient color must be a finite non-negative RGB triple.')
        if self.ambient_ground_color is not None:
            gcolor = np.asarray(self.ambient_ground_color, dtype=np.float64)
            if gcolor.shape != (3,) or not np.isfinite(gcolor).all() or np.any(gcolor < 0):
                raise ValueError('Ambient ground color must be a finite non-negative RGB triple.')
        if self.environment is not None:
            self.environment.validate()
        for light in self.lights:
            light.validate()


@dataclass(frozen=True)
class EngineSettings:
    base_blur_sigma: float = 18.0
    detail_power: float = 0.90
    ratio_min: float = 0.25
    ratio_max: float = 4.0
    color_ratio_min: float = 0.15
    color_ratio_max: float = 3.5
    specular_scale: float = 1.0
    shadow_steps: int = 24
    shadow_bias_metric: float = 0.015
    shadow_bias_relative: float = 0.003
    shadow_max_thickness: Optional[float] = None
    shadow_adaptive_sampling: bool = True
    shadow_smooth_passes: int = 1
    quality_mode: RenderQualityMode = 'standard'
    preview_max_side: Optional[int] = None
    preview_upsample: bool = True
    min_geometry_confidence: float = 0.15
    output_exposure: float = 1.0
    edge_aware_base: bool = True
    renderer_mode: RendererMode = 'auto'
    detail_preservation_mode: DetailPreservationMode = 'baseline'
    detail_lock_strength: float = 0.95
    preserve_source_lighting: bool = False
    lighting_smooth_sigma: float = 1.2
    normal_smooth_sigma: float = 0.0
    source_shading_blend: float = 0.0
    auto_sh_source: bool = False
    indirect_bounce_strength: float = 0.0
    indirect_bounce_radius: float = 24.0
    filmic_highlight_rolloff: bool = False
    refine_depth_edges: bool = False
    atmosphere_density: float = 0.0
    atmosphere_color: Optional[Tuple[float, float, float]] = None
    color_transfer_strength: float = 1.0
    background_light_influence: float = 1.0
    environment_light_influence: float = 1.0
    real_image_preservation: float = 0.0
    highlight_relocation_strength: float = 0.0
    wavelet_levels: int = 4
    wavelet_mid_blend: float = 0.55
    operation: RelightOperation = 'replace'
    source_exposure_scale: float = 1.0
    source_white_balance: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    protect_sky: bool = True
    volumetric_beam_strength: float = 0.0
    shadow_width_aware: bool = False
    shadow_half_res: bool = False
    semantic_material_priors: bool = False
    albedo_guided_source_shading: bool = False
    ssao_strength: float = 0.0
    indirect_specular_strength: float = 0.0
    preserve_fill_balance: float = 0.0
    lossless_ratio_detail: bool = False

    @classmethod
    def real_preserving(cls, **overrides) -> 'EngineSettings':
        """Return calibrated Tier 1 + Wavelet identity-lock settings that preserve the real photograph."""
        defaults = {
            'detail_preservation_mode': 'wavelet',
            'color_transfer_strength': 0.50,
            'background_light_influence': 0.25,
            'environment_light_influence': 0.25,
            'real_image_preservation': 0.45,
            'highlight_relocation_strength': 0.75,
            'auto_sh_source': True,
            'source_shading_blend': 0.35,
            'filmic_highlight_rolloff': True,
            'lighting_smooth_sigma': 1.5,
            'wavelet_levels': 4,
            'wavelet_mid_blend': 0.55,
            'shadow_width_aware': True,
        }
        defaults.update(overrides)
        settings = cls(**defaults)
        settings.validate()
        return settings

    @classmethod
    def studio_3d(cls, **overrides) -> 'EngineSettings':
        """Return whole-scene 3D studio light transport preset with global bounce, SSAO, and material response."""
        defaults = {
            'detail_preservation_mode': 'wavelet',
            'color_transfer_strength': 0.72,
            'background_light_influence': 0.90,
            'environment_light_influence': 0.95,
            'real_image_preservation': 0.32,
            'highlight_relocation_strength': 0.75,
            'indirect_bounce_strength': 0.55,
            'indirect_bounce_radius': 32.0,
            'indirect_specular_strength': 0.45,
            'ssao_strength': 0.42,
            'volumetric_beam_strength': 0.08,
            'auto_sh_source': True,
            'source_shading_blend': 0.45,
            'albedo_guided_source_shading': True,
            'semantic_material_priors': True,
            'filmic_highlight_rolloff': True,
            'refine_depth_edges': True,
            'lighting_smooth_sigma': 1.5,
            'shadow_steps': 32,
            'shadow_width_aware': True,
            'shadow_half_res': True,
            'wavelet_levels': 4,
            'wavelet_mid_blend': 0.55,
            'protect_sky': True,
            'preserve_fill_balance': 0.55,
            'lossless_ratio_detail': True,
        }
        defaults.update(overrides)
        settings = cls(**defaults)
        settings.validate()
        return settings

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
            'detail_lock_strength': self.detail_lock_strength,
            'lighting_smooth_sigma': self.lighting_smooth_sigma,
            'normal_smooth_sigma': self.normal_smooth_sigma,
            'source_shading_blend': self.source_shading_blend,
            'indirect_bounce_strength': self.indirect_bounce_strength,
            'indirect_bounce_radius': self.indirect_bounce_radius,
            'atmosphere_density': self.atmosphere_density,
            'color_transfer_strength': self.color_transfer_strength,
            'background_light_influence': self.background_light_influence,
            'real_image_preservation': self.real_image_preservation,
            'highlight_relocation_strength': self.highlight_relocation_strength,
            'wavelet_mid_blend': self.wavelet_mid_blend,
            'source_exposure_scale': self.source_exposure_scale,
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
        if self.shadow_max_thickness is not None:
            _finite_scalar('shadow_max_thickness', self.shadow_max_thickness)
            if self.shadow_max_thickness <= 0:
                raise ValueError('shadow_max_thickness must be positive when set.')
        if not isinstance(self.shadow_steps, int) or self.shadow_steps < 2:
            raise ValueError('shadow_steps must be an integer >= 2.')
        if not isinstance(self.shadow_adaptive_sampling, bool):
            raise ValueError('shadow_adaptive_sampling must be a boolean.')
        if not isinstance(self.shadow_smooth_passes, int) or self.shadow_smooth_passes < 1:
            raise ValueError('shadow_smooth_passes must be an integer >= 1.')
        if self.quality_mode not in ('standard', 'preview', 'final'):
            raise ValueError("quality_mode must be 'standard', 'preview', or 'final'.")
        if self.preview_max_side is not None:
            if not isinstance(self.preview_max_side, int) or self.preview_max_side < 32:
                raise ValueError('preview_max_side must be an integer >= 32 when set.')
        if not isinstance(self.preview_upsample, bool):
            raise ValueError('preview_upsample must be a boolean.')
        if not 0 <= self.min_geometry_confidence <= 1:
            raise ValueError('min_geometry_confidence must be in [0,1].')
        if not isinstance(self.edge_aware_base, bool):
            raise ValueError('edge_aware_base must be a boolean.')
        if self.renderer_mode not in ('auto', 'ratio', 'intrinsic', 'hybrid_neural'):
            raise ValueError("renderer_mode must be 'auto', 'ratio', 'intrinsic', or 'hybrid_neural'.")
        if self.detail_preservation_mode not in ('baseline', 'f3', 'wavelet'):
            raise ValueError("detail_preservation_mode must be 'baseline', 'f3', or 'wavelet'.")
        if not 0.0 <= self.detail_lock_strength <= 1.0:
            raise ValueError('detail_lock_strength must be in [0,1].')
        if not isinstance(self.preserve_source_lighting, bool):
            raise ValueError('preserve_source_lighting must be a boolean.')
        if self.lighting_smooth_sigma < 0:
            raise ValueError('lighting_smooth_sigma must be non-negative.')
        if self.normal_smooth_sigma < 0:
            raise ValueError('normal_smooth_sigma must be non-negative.')
        if not 0.0 <= self.source_shading_blend <= 1.0:
            raise ValueError('source_shading_blend must be in [0,1].')
        if not isinstance(self.auto_sh_source, bool):
            raise ValueError('auto_sh_source must be a boolean.')
        if self.indirect_bounce_strength < 0.0:
            raise ValueError('indirect_bounce_strength must be non-negative.')
        if self.indirect_bounce_radius <= 0.0:
            raise ValueError('indirect_bounce_radius must be positive.')
        if not isinstance(self.filmic_highlight_rolloff, bool):
            raise ValueError('filmic_highlight_rolloff must be a boolean.')
        if not isinstance(self.refine_depth_edges, bool):
            raise ValueError('refine_depth_edges must be a boolean.')
        if self.atmosphere_density < 0.0:
            raise ValueError('atmosphere_density must be non-negative.')
        if self.atmosphere_color is not None:
            acolor = np.asarray(self.atmosphere_color, dtype=np.float64)
            if acolor.shape != (3,) or not np.isfinite(acolor).all() or np.any(acolor < 0):
                raise ValueError('atmosphere_color must be a finite non-negative RGB triple.')
        for unit_name, unit_val in (
            ('color_transfer_strength', self.color_transfer_strength),
            ('background_light_influence', self.background_light_influence),
            ('environment_light_influence', self.environment_light_influence),
            ('real_image_preservation', self.real_image_preservation),
            ('highlight_relocation_strength', self.highlight_relocation_strength),
            ('wavelet_mid_blend', self.wavelet_mid_blend),
            ('volumetric_beam_strength', self.volumetric_beam_strength),
            ('ssao_strength', self.ssao_strength),
            ('indirect_specular_strength', self.indirect_specular_strength),
            ('preserve_fill_balance', self.preserve_fill_balance),
        ):
            if not 0.0 <= unit_val <= 1.0:
                raise ValueError(f'{unit_name} must be in [0,1].')
        if self.operation not in ('replace', 'add'):
            raise ValueError("operation must be 'replace' or 'add'.")
        if self.source_exposure_scale <= 0.0:
            raise ValueError('source_exposure_scale must be positive.')
        wb = np.asarray(self.source_white_balance, dtype=np.float64)
        if wb.shape != (3,) or not np.isfinite(wb).all() or np.any(wb <= 0.0):
            raise ValueError('source_white_balance must be a finite positive 3-vector.')
        if not isinstance(self.wavelet_levels, int) or not (1 <= self.wavelet_levels <= 6):
            raise ValueError('wavelet_levels must be an integer in [1, 6].')
        for bname, bval in (
            ('protect_sky', self.protect_sky),
            ('shadow_width_aware', self.shadow_width_aware),
            ('shadow_half_res', self.shadow_half_res),
            ('semantic_material_priors', self.semantic_material_priors),
            ('albedo_guided_source_shading', self.albedo_guided_source_shading),
            ('lossless_ratio_detail', self.lossless_ratio_detail),
        ):
            if not isinstance(bval, bool):
                raise ValueError(f'{bname} must be a boolean.')


@dataclass
class RelightResult:
    rgb_srgb: np.ndarray
    normal_map: np.ndarray
    diffuse_ratio: np.ndarray
    shadow_visibility: np.ndarray
    geometry_confidence: np.ndarray
    metrics: dict
