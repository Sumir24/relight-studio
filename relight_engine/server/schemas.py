"""Pydantic request/response schemas for the Relight Studio HTTP API."""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class SourceCalibrationSchema(BaseModel):
    name: str = "user_calibrated_source_key"
    key_direction: tuple[float, float, float]
    key_strength: float = Field(1.0, ge=0.0)
    key_temperature_kelvin: float = Field(5600.0, ge=1000.0, le=40000.0)
    ambient_strength: float = Field(0.15, ge=0.0)
    ambient_temperature_kelvin: float = Field(6500.0, ge=1000.0, le=40000.0)
    casts_shadows: bool = False


class LightSchema(BaseModel):
    name: Optional[str] = None
    kind: Literal["directional", "point", "area"]
    strength: float = Field(1.0, ge=0.0)
    color: Optional[tuple[float, float, float]] = None
    color_temperature_kelvin: Optional[float] = Field(None, ge=1000.0, le=40000.0)
    direction: Optional[tuple[float, float, float]] = None
    position: Optional[tuple[float, float, float]] = None
    area_size: tuple[float, float] = (0.5, 0.5)
    samples: int = Field(4, ge=1, le=64)
    casts_shadows: bool = True
    wrap: float = Field(0.0, ge=0.0, le=1.0)
    radius: float = Field(0.0, ge=0.0)


class EnvironmentLightSchema(BaseModel):
    sky_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    horizon_color: tuple[float, float, float] = (0.75, 0.78, 0.85)
    ground_color: tuple[float, float, float] = (0.20, 0.20, 0.22)
    strength: float = Field(0.0, ge=0.0)
    key_azimuth_deg: float = 0.0
    key_elevation_deg: float = 35.0
    key_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    key_strength: float = Field(0.0, ge=0.0)
    key_sharpness: float = Field(12.0, gt=0.0)
    env_map_strength: float = Field(1.0, ge=0.0)
    diffuse_strength: float = Field(1.0, ge=0.0)
    specular_strength: float = Field(1.0, ge=0.0)
    rotation_deg: float = 0.0
    env_map_path: Optional[str] = None


class TargetRigSchema(BaseModel):
    ambient_strength: float = Field(0.15, ge=0.0)
    ambient_color: Optional[tuple[float, float, float]] = None
    ambient_temperature_kelvin: Optional[float] = Field(None, ge=1000.0, le=40000.0)
    ambient_ground_color: Optional[tuple[float, float, float]] = None
    ambient_ground_temperature_kelvin: Optional[float] = Field(None, ge=1000.0, le=40000.0)
    lights: list[LightSchema] = Field(default_factory=list)
    environment: Optional[EnvironmentLightSchema] = None


class MaterialAssignmentSchema(BaseModel):
    name: str
    mask_key: str
    roughness: float = Field(..., ge=0.0, le=1.0)
    metalness: float = Field(0.0, ge=0.0, le=1.0)
    specular_level: float = Field(0.04, ge=0.0, le=1.0)
    specular_color: Optional[tuple[float, float, float]] = None
    clearcoat: float = Field(0.0, ge=0.0, le=1.0)
    clearcoat_roughness: float = Field(0.15, ge=0.0, le=1.0)
    subsurface: float = Field(0.0, ge=0.0, le=1.0)
    anisotropy: float = Field(0.0, ge=0.0, le=1.0)
    anisotropy_angle: float = 0.0
    blend: float = Field(1.0, ge=0.0, le=1.0)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class EngineSettingsSchema(BaseModel):
    base_blur_sigma: float = Field(18.0, gt=0.0)
    detail_power: float = Field(1.0, ge=0.0)
    ratio_min: float = Field(0.25, gt=0.0)
    ratio_max: float = Field(4.0, gt=0.0)
    color_ratio_min: float = Field(0.15, gt=0.0)
    color_ratio_max: float = Field(3.5, gt=0.0)
    specular_scale: float = Field(1.0, ge=0.0)
    shadow_steps: int = Field(24, ge=2, le=128)
    shadow_bias_metric: float = Field(0.015, ge=0.0)
    shadow_bias_relative: float = Field(0.003, ge=0.0)
    shadow_max_thickness: Optional[float] = Field(None, gt=0.0)
    shadow_adaptive_sampling: bool = True
    shadow_smooth_passes: int = Field(1, ge=1, le=6)
    quality_mode: Literal["standard", "preview", "final"] = "standard"
    preview_max_side: Optional[int] = Field(None, ge=32, le=4096)
    preview_upsample: bool = True
    min_geometry_confidence: float = Field(0.15, ge=0.0, le=1.0)
    output_exposure: float = Field(1.0, ge=0.0)
    edge_aware_base: bool = True
    renderer_mode: Literal["auto", "ratio", "intrinsic", "hybrid_neural"] = "auto"
    detail_preservation_mode: Literal["baseline", "f3", "wavelet"] = "baseline"
    detail_lock_strength: float = Field(0.95, ge=0.0, le=1.0)
    preserve_source_lighting: bool = False
    lighting_smooth_sigma: float = Field(1.2, ge=0.0)
    normal_smooth_sigma: float = Field(0.0, ge=0.0)
    source_shading_blend: float = Field(0.0, ge=0.0, le=1.0)
    auto_sh_source: bool = False
    indirect_bounce_strength: float = Field(0.0, ge=0.0)
    indirect_bounce_radius: float = Field(24.0, gt=0.0)
    filmic_highlight_rolloff: bool = False
    refine_depth_edges: bool = False
    atmosphere_density: float = Field(0.0, ge=0.0)
    atmosphere_color: Optional[tuple[float, float, float]] = None
    color_transfer_strength: float = Field(1.0, ge=0.0, le=1.0)
    background_light_influence: float = Field(1.0, ge=0.0, le=1.0)
    real_image_preservation: float = Field(0.0, ge=0.0, le=1.0)
    highlight_relocation_strength: float = Field(0.0, ge=0.0, le=1.0)
    wavelet_levels: int = Field(4, ge=1, le=6)
    wavelet_mid_blend: float = Field(0.55, ge=0.0, le=1.0)


class RelightRequest(BaseModel):
    source_calibration: SourceCalibrationSchema
    target_rig: TargetRigSchema
    material_assignments: list[MaterialAssignmentSchema] = Field(default_factory=list)
    settings: EngineSettingsSchema = Field(default_factory=EngineSettingsSchema)
    require_geometry_ready: bool = True
    relighter_backend: Optional[str] = None
    relighter_options: dict[str, Any] = Field(default_factory=dict)
    preview_max_side: Optional[int] = Field(None, ge=32, le=4096)


class HealthResponse(BaseModel):
    status: str
    device: str
    cached_scenes: int
    total_scenes: int
    active_jobs: int
    available_backends: dict[str, tuple[str, ...]]


class AnalyzeResponse(BaseModel):
    job_id: str
    scene_id: str
    status: str
    status_url: str
    scene_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    scene_id: str
    status: Literal["queued", "running", "ready", "failed"]
    stage: str
    progress: float
    error: Optional[str] = None
    scene_url: Optional[str] = None


class SceneSummaryResponse(BaseModel):
    scene_id: str
    resolution: list[int]
    depth_kind: str
    semantic_labels: list[str]
    instance_masks: list[str]
    camera: dict[str, Any]
    artifacts: dict[str, str]
    metrics: dict[str, Any]


class RelightResponse(BaseModel):
    scene_id: str
    render_id: str
    render_url: str
    report_url: str
    elapsed_ms: float
    resolution: list[int]
    metrics: dict[str, Any]
