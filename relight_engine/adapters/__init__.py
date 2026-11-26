from .depth_anything_hf import HuggingFaceDepthAnythingV2, DepthPrediction, normalize_relative_depth, relative_depth_polarity_candidates
from .scene_factory import SemanticPrediction, build_scene_buffers
from .grounded_sam2 import HuggingFaceGroundedSAM2, SceneSegmentationPrediction, DetectionRecord
from ..scene_policy import SemanticRoutingPolicy, compose_scene_masks, semantic_key
from .florence2_discovery import Florence2SceneDiscovery, SceneDiscoveryResult
from .camera import CameraCalibration, camera_from_horizontal_fov
from .diffusers_normals import DiffusersNormalAdapter
from .diffusers_intrinsics import DiffusersIntrinsicAdapter
from .ic_light_relighter import ICLightRelighter
from .flux_iclight_relighter import FluxICLightRelighter
from .protocols import (
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
from .registry import (
    available_backends,
    create_intrinsic_backend,
    create_normal_backend,
    create_relighter_backend,
    register_intrinsic_backend,
    register_normal_backend,
    register_relighter_backend,
)

__all__ = [
    'HuggingFaceDepthAnythingV2', 'DepthPrediction', 'normalize_relative_depth', 'relative_depth_polarity_candidates', 'SemanticPrediction', 'build_scene_buffers',
    'HuggingFaceGroundedSAM2', 'SceneSegmentationPrediction', 'DetectionRecord',
    'SemanticRoutingPolicy', 'compose_scene_masks', 'semantic_key',
    'Florence2SceneDiscovery', 'SceneDiscoveryResult',
    'CameraCalibration', 'camera_from_horizontal_fov',
    'DiffusersNormalAdapter', 'DiffusersIntrinsicAdapter', 'ICLightRelighter', 'FluxICLightRelighter',
    'DepthEstimator', 'NormalEstimator', 'NormalPrediction',
    'IntrinsicEstimator', 'IntrinsicPrediction',
    'NeuralRelighter', 'NeuralRelightPrediction',
    'SceneDiscoverer', 'SceneSegmenter',
    'register_normal_backend', 'create_normal_backend',
    'register_intrinsic_backend', 'create_intrinsic_backend',
    'register_relighter_backend', 'create_relighter_backend',
    'available_backends',
]
