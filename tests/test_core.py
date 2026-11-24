import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from relight_engine import CameraModel, SceneBuffers, Light, LightingRig, EngineSettings, relight
from relight_engine.geometry import backproject_depth, estimate_normals
from relight_engine.lighting import evaluate_rig
from relight_engine.shadows import ray_visibility
from relight_engine.adapters import HuggingFaceDepthAnythingV2, DepthPrediction, SemanticPrediction, build_scene_buffers


def make_plane(h=48, w=64, slope=0.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x_norm = (xx - w / 2) / 80.0
    depth = 4.0 / (1.0 + slope * x_norm)
    rgb = np.full((h, w, 3), 0.45, np.float32)
    conf = np.ones((h, w), np.float32)
    mask = np.ones((h, w), np.float32)
    scene = SceneBuffers(rgb, depth.astype(np.float32), conf, mask, 'metric',
                         roughness=np.full((h, w), 0.55, np.float32),
                         metalness=np.zeros((h, w), np.float32))
    camera = CameraModel(80, 80, w / 2, h / 2)
    return scene, camera


def test_front_plane_normals_face_camera():
    scene, camera = make_plane()
    p = backproject_depth(scene.depth, camera)
    n, c = estimate_normals(p, scene.depth, scene.depth_confidence)
    assert np.mean(n[..., 2]) < -0.999
    assert np.mean(c) > 0.99


def test_slope_changes_normal_and_light_response():
    scene, camera = make_plane(slope=1.8)
    p = backproject_depth(scene.depth, camera)
    n, c = estimate_normals(p, scene.depth, scene.depth_confidence)
    assert np.std(n[..., 0]) < 0.02
    assert abs(float(np.mean(n[..., 0]))) > 0.01
    light_right = LightingRig((1,1,1), 0.05,
                              (Light('right', 'directional', strength=1.0, direction=(1, 0, -1)),))
    light_left = LightingRig((1,1,1), 0.05,
                             (Light('left', 'directional', strength=1.0, direction=(-1, 0, -1)),))
    d_r, _, _ = evaluate_rig(scene, camera, p, n, light_right, EngineSettings())
    d_l, _, _ = evaluate_rig(scene, camera, p, n, light_left, EngineSettings())
    assert abs(float(np.mean(d_r[..., 0] - d_l[..., 0]))) > 1e-3


def test_same_source_and_target_is_near_noop():
    scene, camera = make_plane()
    rig = LightingRig((1,1,1), 0.15,
                      (Light('key', 'directional', strength=0.7, direction=(0.2, 0.3, -1)),))
    result = relight(scene, camera, rig, rig)
    assert np.max(np.abs(result.rgb_srgb - scene.rgb_srgb)) < 2e-4


def test_light_is_geometry_dependent():
    flat, camera = make_plane(slope=0.0)
    tilted, _ = make_plane(slope=2.4)
    light = LightingRig((1,1,1), 0.02,
                        (Light('key', 'directional', strength=1.0, direction=(0.9, 0.0, -0.3)),))
    def mean_direct(scene):
        p = backproject_depth(scene.depth, camera)
        n, _ = estimate_normals(p, scene.depth, scene.depth_confidence)
        d, _, _ = evaluate_rig(scene, camera, p, n, light, EngineSettings())
        return float(np.mean(d))
    assert abs(mean_direct(flat) - mean_direct(tilted)) > 1e-3




def test_noop_preserves_highlights_exactly():
    scene, camera = make_plane()
    scene.rgb_srgb[:] = 0.98
    rig = LightingRig((1,1,1), 0.15,
                      (Light('key', 'directional', strength=0.7, direction=(0.2, 0.3, -1)),))
    result = relight(scene, camera, rig, rig)
    assert np.max(np.abs(result.rgb_srgb - scene.rgb_srgb)) < 2e-4


def test_rough_material_has_broader_lower_specular_peak():
    scene, camera = make_plane()
    scene.roughness[:] = 0.12
    p = backproject_depth(scene.depth, camera)
    n, _ = estimate_normals(p, scene.depth, scene.depth_confidence)
    rig = LightingRig((1,1,1), 0.0,
                      (Light('key', 'directional', strength=1.0, direction=(0.0, 0.0, -1), casts_shadows=False),))
    _, spec_smooth, _ = evaluate_rig(scene, camera, p, n, rig, EngineSettings())
    scene.roughness[:] = 0.85
    _, spec_rough, _ = evaluate_rig(scene, camera, p, n, rig, EngineSettings())
    assert float(np.max(spec_smooth)) > float(np.max(spec_rough))


def test_depth_ray_finds_a_nearer_occluder():
    h, w = 64, 64
    camera = CameraModel(50, 50, w / 2, h / 2)
    depth = np.full((h, w), 5.0, np.float32)
    depth[29:36, 19:26] = 2.0  # foreground object in front of the background plane
    points = backproject_depth(depth, camera)
    n = np.zeros_like(points)
    n[..., 2] = -1
    to_light = np.zeros_like(points)
    to_light[:] = np.array([-1.0, 0.0, -0.5], np.float32)
    max_dist = np.full((h, w), 5.0, np.float32)
    visibility = ray_visibility(points, n, depth, camera, to_light, max_dist, 'metric', EngineSettings(shadow_steps=18))
    # Receiver to the right of the occluder sees it along the ray to this light.
    assert visibility[32, 40] < 0.5



def test_relative_depth_adapter_requires_explicit_polarity():
    try:
        HuggingFaceDepthAnythingV2(output_kind='relative')
    except ValueError:
        return
    raise AssertionError('Relative depth polarity must be explicitly configured.')

def test_metric_depth_adapter_requires_license_review():
    try:
        HuggingFaceDepthAnythingV2(model_id='depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf',
                                   output_kind='metric')
    except ValueError:
        return
    raise AssertionError('Metric model load should be blocked pending exact checkpoint license review.')

def test_scene_factory_rejects_resolution_mismatch():
    rgb = np.zeros((12, 16, 3), np.float32)
    depth = DepthPrediction(np.ones((8, 8), np.float32), np.ones((8, 8), np.float32), 'relative')
    sem = SemanticPrediction(np.ones((12, 16), np.float32))
    try:
        build_scene_buffers(rgb, depth, sem)
    except ValueError:
        return
    raise AssertionError('Scene factory must reject unregistered depth maps.')



def test_moving_point_light_moves_illumination_centroid():
    scene, camera = make_plane()
    scene.roughness[:] = 0.8
    p = backproject_depth(scene.depth, camera)
    n, _ = estimate_normals(p, scene.depth, scene.depth_confidence)
    def centroid_x(light_x):
        rig = LightingRig((1,1,1), 0.01,
                          (Light('point', 'point', strength=30.0, position=(light_x, 0.0, 1.0),
                                 casts_shadows=False),))
        d, _, _ = evaluate_rig(scene, camera, p, n, rig, EngineSettings())
        weights = d[..., 0]
        xs = np.arange(weights.shape[1], dtype=np.float32)[None, :]
        return float(np.sum(weights * xs) / np.sum(weights))
    left = centroid_x(-1.0)
    right = centroid_x(1.0)
    assert right > left + 2.0

def test_invalid_shapes_fail_fast():
    scene, _ = make_plane()
    scene.depth_confidence = np.ones((3,3), np.float32)
    try:
        scene.validate()
    except ValueError:
        return
    raise AssertionError('Expected validation error for mismatched depth confidence shape.')

if __name__ == '__main__':
    for fn in [test_front_plane_normals_face_camera, test_slope_changes_normal_and_light_response,
               test_same_source_and_target_is_near_noop, test_light_is_geometry_dependent,
               test_noop_preserves_highlights_exactly, test_rough_material_has_broader_lower_specular_peak,
               test_depth_ray_finds_a_nearer_occluder, test_relative_depth_adapter_requires_explicit_polarity,
               test_metric_depth_adapter_requires_license_review, test_scene_factory_rejects_resolution_mismatch,
               test_moving_point_light_moves_illumination_centroid, test_invalid_shapes_fail_fast]:
        fn()
        print('PASS', fn.__name__)
