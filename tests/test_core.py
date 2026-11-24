import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
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



def test_coloured_conductor_f0_changes_specular_reflection_tint():
    scene, camera = make_plane()
    scene.roughness[:] = 0.2
    scene.metalness[:] = 1.0
    p = backproject_depth(scene.depth, camera)
    n, _ = estimate_normals(p, scene.depth, scene.depth_confidence)
    rig = LightingRig(
        (1, 1, 1), 0.0,
        (Light("white_key", "directional", strength=1.0,
               direction=(0.0, 0.0, -1.0), casts_shadows=False),),
    )

    scene.specular_color = np.zeros((*scene.shape, 3), dtype=np.float32)
    scene.specular_color[:] = (0.82, 0.12, 0.08)
    _, red_conductor, _ = evaluate_rig(
        scene, camera, p, n, rig, EngineSettings()
    )

    scene.specular_color[:] = (0.08, 0.18, 0.88)
    _, blue_conductor, _ = evaluate_rig(
        scene, camera, p, n, rig, EngineSettings()
    )

    red_mean = np.mean(red_conductor, axis=(0, 1))
    blue_mean = np.mean(blue_conductor, axis=(0, 1))
    assert red_mean[0] > red_mean[2] * 4.0
    assert blue_mean[2] > blue_mean[0] * 4.0


def test_zero_material_confidence_blends_supplied_maps_to_declared_priors():
    scene, camera = make_plane()
    scene.roughness[:] = 0.12
    scene.metalness[:] = 1.0
    scene.specular_level = np.full(scene.shape, 0.9, dtype=np.float32)
    scene.specular_color = np.full((*scene.shape, 3), (0.9, 0.1, 0.1), dtype=np.float32)
    scene.material_confidence = np.zeros(scene.shape, dtype=np.float32)

    prior_scene, _ = make_plane()
    prior_scene.roughness[:] = 0.75
    prior_scene.metalness[:] = 0.0
    p = backproject_depth(scene.depth, camera)
    n, _ = estimate_normals(p, scene.depth, scene.depth_confidence)
    rig = LightingRig(
        (1, 1, 1), 0.0,
        (Light("key", "directional", strength=1.0,
               direction=(0.0, 0.0, -1.0), casts_shadows=False),),
    )

    supplied_diffuse, supplied_specular, _ = evaluate_rig(
        scene, camera, p, n, rig, EngineSettings()
    )
    prior_diffuse, prior_specular, _ = evaluate_rig(
        prior_scene, camera, p, n, rig, EngineSettings()
    )

    assert np.allclose(supplied_diffuse, prior_diffuse, atol=1e-7)
    assert np.allclose(supplied_specular, prior_specular, atol=1e-7)


def test_scene_contract_rejects_invalid_material_map_shapes_and_values():
    scene, _ = make_plane()
    scene.specular_color = np.zeros((*scene.shape, 4), dtype=np.float32)
    try:
        scene.validate()
    except ValueError as exc:
        assert "specular_color" in str(exc)
    else:
        raise AssertionError("Expected invalid RGB specular-colour map shape to fail.")

    scene, _ = make_plane()
    scene.material_confidence = np.full((3, 3), 0.5, dtype=np.float32)
    try:
        scene.validate()
    except ValueError as exc:
        assert "material_confidence" in str(exc)
    else:
        raise AssertionError("Expected invalid material-confidence shape to fail.")



def test_scene_factory_preserves_material_maps_confidence_and_provenance():
    h, w = 10, 12
    rgb = np.full((h, w, 3), 0.4, np.float32)
    depth = DepthPrediction(
        np.full((h, w), 2.0, np.float32),
        np.ones((h, w), np.float32),
        "relative",
    )
    semantics = SemanticPrediction(
        lightable_mask=np.ones((h, w), np.float32),
        roughness=np.full((h, w), 0.35, np.float32),
        metalness=np.full((h, w), 0.9, np.float32),
        specular_level=np.full((h, w), 0.04, np.float32),
        specular_color=np.broadcast_to(
            np.array([0.8, 0.4, 0.2], np.float32), (h, w, 3)
        ).copy(),
        material_confidence=np.full((h, w), 0.75, np.float32),
        material_metadata={
            "specular_color": {"source": "user_authored", "calibrated": False}
        },
    )

    scene = build_scene_buffers(rgb, depth, semantics)

    assert scene.specular_color is not None
    assert np.allclose(scene.specular_color[0, 0], [0.8, 0.4, 0.2])
    assert scene.material_confidence[0, 0] == 0.75
    assert scene.material_metadata["specular_color"]["source"] == "user_authored"
    scene.validate()


def test_directional_shadow_ray_catches_near_occluder():
    from relight_engine.shadows import scene_shadow_visibility

    h, w = 48, 64
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    depth = np.full((h, w), 5.0, dtype=np.float32)
    # Near thin occluder step ~0.35 units closer to the camera than the background
    depth[18:30, 22:26] = 4.65
    scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.5, dtype=np.float32),
        depth=depth,
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
    )
    points = backproject_depth(depth, camera)
    normals = np.zeros_like(points)
    normals[..., 2] = -1.0
    to_light = np.zeros_like(points)
    to_light[:] = np.array([-1.0, 0.0, -0.45], dtype=np.float32)
    visibility = scene_shadow_visibility(
        scene, camera, points, normals, None, to_light, EngineSettings(shadow_steps=12)
    )
    # Receiver just to the right of the ridge must catch the near shadow
    assert float(visibility[24, 30]) < 0.5


def test_area_light_aims_toward_scene_centroid():
    from relight_engine.lighting import _area_samples

    pos = (3.0, 0.0, 3.0)
    target = np.array([0.0, 0.0, 4.5], dtype=np.float32)
    light = Light("area", "area", position=pos, area_size=(2.0, 2.0), samples=4)
    samples = _area_samples(light, target_point=target)
    aim = target - np.asarray(pos, dtype=np.float32)
    for sample in samples:
        assert abs(float(np.dot(sample - np.asarray(pos, dtype=np.float32), aim))) < 1e-4


def test_edge_aware_compositor_reduces_halo_across_depth_boundary():
    h, w = 40, 64
    rgb = np.full((h, w, 3), 0.05, dtype=np.float32)
    rgb[:, : w // 2] = 0.85
    depth = np.full((h, w), 5.0, dtype=np.float32)
    depth[:, : w // 2] = 1.5
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=depth,
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    source = LightingRig(ambient_strength=0.1, lights=(
        Light("src", "directional", strength=0.4, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    target = LightingRig(ambient_strength=0.1, lights=(
        Light("tgt", "directional", strength=1.4, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    result_edge = relight(
        scene, camera, source, target,
        EngineSettings(base_blur_sigma=12.0, edge_aware_base=True),
    )
    result_gauss = relight(
        scene, camera, source, target,
        EngineSettings(base_blur_sigma=12.0, edge_aware_base=False),
    )
    # Dark side just past the step (away from foreground leak) should not suffer a bright/dark halo jump
    edge_band_edge = result_edge.rgb_srgb[:, w // 2 + 1 : w // 2 + 5, 0]
    far_bg_edge = result_edge.rgb_srgb[:, -8:, 0]
    edge_band_gauss = result_gauss.rgb_srgb[:, w // 2 + 1 : w // 2 + 5, 0]
    far_bg_gauss = result_gauss.rgb_srgb[:, -8:, 0]
    halo_edge = float(np.max(np.abs(np.mean(edge_band_edge) - np.mean(far_bg_edge))))
    halo_gauss = float(np.max(np.abs(np.mean(edge_band_gauss) - np.mean(far_bg_gauss))))
    assert halo_edge < halo_gauss * 0.25


def test_ggx_direct_lighting_schlick_k_uses_roughness():
    from relight_engine.lighting import _ggx_specular

    normals = np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32)
    view = np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32)
    light_dir = np.array([[[0.6, 0.0, -0.8]]], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir, axis=-1, keepdims=True)
    roughness = np.array([[0.5]], dtype=np.float32)
    metalness = np.array([[0.0]], dtype=np.float32)
    spec_level = np.array([[0.04]], dtype=np.float32)
    spec_color = np.full((1, 1, 3), 0.65, dtype=np.float32)

    spec = _ggx_specular(normals, view, light_dir, roughness, metalness, spec_level, spec_color)

    # Compute reference UE4 direct-lighting Schlick-GGX with k = (roughness + 1)^2 / 8
    ndotl = float(np.sum(normals * light_dir))
    ndotv = 1.0
    h = view + light_dir
    h /= np.linalg.norm(h)
    ndoth = float(np.sum(normals * h))
    vdoth = float(np.sum(view * h))
    alpha = 0.5 ** 2
    alpha2 = alpha ** 2
    d_term = alpha2 / (np.pi * (ndoth * ndoth * (alpha2 - 1.0) + 1.0) ** 2)
    k = (0.5 + 1.0) ** 2 / 8.0
    g_term = (ndotv / (ndotv * (1.0 - k) + k)) * (ndotl / (ndotl * (1.0 - k) + k))
    f0 = 0.04
    fresnel = f0 + (1.0 - f0) * (1.0 - vdoth) ** 5
    expected = (d_term * g_term * ndotl) / (4.0 * ndotv * ndotl) * fresnel
    assert float(spec[0, 0, 0]) == pytest.approx(expected, rel=1e-4)
