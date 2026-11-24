from __future__ import annotations
import numpy as np
from .contracts import CameraModel, SceneBuffers, LightingRig, EngineSettings, RelightResult
from .geometry import prepare_geometry
from .lighting import evaluate_rig
from .composite import compose_source_preserving
from .color import srgb_to_linear

def relight(scene: SceneBuffers, camera: CameraModel, source_rig: LightingRig,
            target_rig: LightingRig, settings: EngineSettings | None = None) -> RelightResult:
    """Relight an image using shared 2.5D geometry and source-to-target shading transfer.

    Requires externally estimated depth and scene masks. Does not fabricate geometry.
    """
    settings = settings or EngineSettings()
    settings.validate()
    scene.validate()
    camera.validate()
    points, normals, geometry_conf = prepare_geometry(scene, camera)
    source_d, source_s, _ = evaluate_rig(scene, camera, points, normals, source_rig, settings)
    target_d, target_s, visibility = evaluate_rig(scene, camera, points, normals, target_rig, settings)
    out, ratio = compose_source_preserving(scene, source_d, target_d, source_s, target_s,
                                           geometry_conf, visibility, settings)
    active = (scene.lightable_mask > 0.05) & (geometry_conf > settings.min_geometry_confidence)
    delta = np.abs(out - scene.rgb_srgb)
    metrics = {
        'mode': '2.5D source-to-target relighting',
        'depth_kind': scene.depth_kind,
        'geometry_valid_fraction': float(np.mean((scene.depth > 0) & (geometry_conf > settings.min_geometry_confidence))),
        'lightable_fraction': float(np.mean(active)),
        'mean_abs_srgb_change_active': float(np.mean(delta[active])) if np.any(active) else 0.0,
        'mean_abs_srgb_change_protected': float(np.mean(delta[~active])) if np.any(~active) else 0.0,
        'mean_target_to_source_shading_ratio': float(np.mean(ratio[active])) if np.any(active) else 1.0,
        'shadow_visibility_mean': float(np.mean(visibility[active])) if np.any(active) else 1.0,
        'neural_scene_estimation_used': bool(scene.metadata.get('neural_scene_estimation_used', False)),
        'limitations': [
            '2.5D depth represents only the visible first surface; hidden surfaces and overhangs are unavailable.',
            'Relighting quality depends on depth, camera calibration, semantic masks, and source-light estimation.',
            'Shadow visibility is approximate screen-space ray marching, not a full 3D mesh shadow map.',
            'Single-image intrinsic decomposition remains ambiguous; source_rig must approximate the original illumination.'
        ]
    }
    return RelightResult(out, normals, ratio, visibility, geometry_conf, metrics)
