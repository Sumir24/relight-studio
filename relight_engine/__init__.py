from .contracts import CameraModel, SceneBuffers, Light, LightingRig, EnvironmentLight, EngineSettings, RelightResult
from .engine import relight, RelightSession, prepare_relight_session

__all__ = ['CameraModel', 'SceneBuffers', 'Light', 'LightingRig', 'EnvironmentLight', 'EngineSettings', 'RelightResult', 'relight', 'RelightSession', 'prepare_relight_session']

from .calibration import (
    SourceIllumination,
    SourceLightCalibration,
    MaterialAssignment,
    MaterialMaps,
    build_material_maps,
    compute_intrinsic_reconstruction_error,
    estimate_source_illumination,
    kelvin_to_linear_rgb,
    refine_source_shading_with_albedo,
    suggest_material_priors_from_semantics,
)

__all__ += [
    'SourceIllumination',
    'SourceLightCalibration',
    'MaterialAssignment',
    'MaterialMaps',
    'build_material_maps',
    'compute_intrinsic_reconstruction_error',
    'estimate_source_illumination',
    'kelvin_to_linear_rgb',
    'refine_source_shading_with_albedo',
    'suggest_material_priors_from_semantics',
]

from .image_io import ALPHA_CONVENTION, ImageInput, load_image, save_image
from .geometry import compute_screen_space_ao, place_point_light_at_uv, place_studio_light
from .quality import evaluate_preservation
from .scene_policy import classify_scene_layers, classify_scene_scale, detect_sky_horizon_mask
__all__ += [
    "ALPHA_CONVENTION",
    "ImageInput",
    "load_image",
    "save_image",
    "evaluate_preservation",
    "compute_screen_space_ao",
    "place_point_light_at_uv",
    "place_studio_light",
    "classify_scene_layers",
    "classify_scene_scale",
    "detect_sky_horizon_mask",
]
