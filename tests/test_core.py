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


def test_ray_visibility_produces_soft_antialiased_penumbra_without_hard_stairsteps():
    h, w = 48, 64
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    depth = np.full((h, w), 5.0, dtype=np.float32)
    depth[16:32, 36:] = 1.8
    points = backproject_depth(depth, camera)
    normals = np.zeros_like(points)
    normals[..., 2] = -1.0
    to_light = np.zeros_like(points)
    to_light[:] = np.array([0.4, 0.8, -0.4], dtype=np.float32)
    max_dist = np.full((h, w), 4.0, dtype=np.float32)
    vis = ray_visibility(
        points, normals, depth, camera, to_light, max_dist, "relative",
        EngineSettings(shadow_steps=16),
    )
    bg_vis = vis[:, :35]
    # Expect both occluded core and soft intermediate penumbra values (not strictly binary 0/1)
    assert np.any(bg_vis < 0.25)
    assert np.any((bg_vis > 0.05) & (bg_vis < 0.95))


def test_intrinsic_mode_lifts_baked_shadow_using_albedo_and_preserves_detail():
    from relight_engine.color import linear_to_srgb

    h, w = 32, 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    albedo_lin = np.full((h, w, 3), 0.55, dtype=np.float32)
    # Fine high-frequency texture modulated on albedo/image
    stripes = 1.0 + 0.12 * np.sin(xx * 1.4)
    baked_shading = np.where(xx < w // 2, 0.18, 0.85).astype(np.float32)
    rgb_lin = np.clip(albedo_lin * baked_shading[..., None] * stripes[..., None], 0.0, 1.0)
    rgb_srgb = linear_to_srgb(rgb_lin)

    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene = SceneBuffers(
        rgb_srgb=rgb_srgb,
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        albedo=linear_to_srgb(albedo_lin),
        source_shading=baked_shading,
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    source = LightingRig(ambient_strength=0.2, lights=(
        Light("src", "directional", strength=0.8, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    target = LightingRig(ambient_strength=0.2, lights=(
        Light("tgt", "directional", strength=0.8, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    res = relight(
        scene, camera, source, target,
        EngineSettings(renderer_mode="intrinsic", base_blur_sigma=4.0, detail_lock_strength=0.85),
    )
    left_mean = float(np.mean(res.rgb_srgb[:, 4 : w // 2 - 4]))
    right_mean = float(np.mean(res.rgb_srgb[:, w // 2 + 4 : -4]))
    # Baked left shadow must lift close to the sunlit right side
    assert abs(left_mean - right_mean) < 0.08
    assert res.metrics["renderer_mode"] == "intrinsic"
    # Fine stripe detail remains preserved on the lifted shadow side
    assert float(np.std(res.rgb_srgb[:, 4 : w // 2 - 4, 0])) > 0.02


def test_intrinsic_mode_is_invariant_to_source_shading_scale():
    from relight_engine.color import linear_to_srgb

    h, w = 24, 32
    albedo_lin = np.full((h, w, 3), 0.50, dtype=np.float32)
    base_shading = np.full((h, w), 0.70, dtype=np.float32)
    rgb_srgb = linear_to_srgb(albedo_lin * base_shading[..., None])

    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig = LightingRig(ambient_strength=0.2, lights=(
        Light("key", "directional", strength=0.8, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    outputs = []
    for k_s in (0.2, 1.0, 5.0):
        scene = SceneBuffers(
            rgb_srgb=rgb_srgb,
            depth=np.full((h, w), 2.0, dtype=np.float32),
            depth_confidence=np.ones((h, w), dtype=np.float32),
            lightable_mask=np.ones((h, w), dtype=np.float32),
            depth_kind="relative",
            normals=normals,
            geometry_confidence=np.ones((h, w), dtype=np.float32),
            albedo=linear_to_srgb(albedo_lin),
            source_shading=base_shading * k_s,
        )
        res = relight(scene, camera, rig, rig, EngineSettings(renderer_mode="intrinsic", base_blur_sigma=4.0))
        outputs.append(res.rgb_srgb)

    # Identical calibrated albedo & target rig must yield identical brightness regardless of source_shading scale k_s
    assert np.allclose(outputs[0], outputs[1], atol=1e-4)
    assert np.allclose(outputs[1], outputs[2], atol=1e-4)
    # Albedo 0.5 * target_diffuse ~1.0 (0.2 + 0.8) = ~0.5 linear -> ~0.735 sRGB
    assert 0.65 < float(np.mean(outputs[1])) < 0.80


def test_intrinsic_mode_supports_non_diffuse_residual_and_missing_source_shading():
    from relight_engine.color import linear_to_srgb

    h, w = 24, 32
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    residual = np.full((h, w, 3), 0.16, dtype=np.float32)
    scene_no_sh = SceneBuffers(
        rgb_srgb=linear_to_srgb(np.full((h, w, 3), 0.40, dtype=np.float32)),
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        albedo=linear_to_srgb(np.full((h, w, 3), 0.40, dtype=np.float32)),
        source_shading=None,
        non_diffuse_residual=residual,
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig = LightingRig(ambient_strength=0.2, lights=(
        Light("key", "directional", strength=0.8, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    res_with_res = relight(scene_no_sh, camera, rig, rig, EngineSettings(renderer_mode="intrinsic"))
    scene_no_res = SceneBuffers(
        rgb_srgb=scene_no_sh.rgb_srgb, depth=scene_no_sh.depth,
        depth_confidence=scene_no_sh.depth_confidence, lightable_mask=scene_no_sh.lightable_mask,
        depth_kind="relative", normals=normals, geometry_confidence=scene_no_sh.geometry_confidence,
        albedo=scene_no_sh.albedo, source_shading=None, non_diffuse_residual=None,
    )
    res_no_res = relight(scene_no_res, camera, rig, rig, EngineSettings(renderer_mode="intrinsic"))
    assert float(np.mean(res_with_res.rgb_srgb)) > float(np.mean(res_no_res.rgb_srgb)) + 0.01


def test_intrinsic_mode_errors_when_albedo_missing():
    scene, camera = make_plane()
    rig = LightingRig(ambient_strength=0.2)
    with pytest.raises(ValueError, match="albedo"):
        relight(scene, camera, rig, rig, EngineSettings(renderer_mode="intrinsic"))


def test_hybrid_neural_mode_locks_high_freq_detail_protected_mask_and_alpha():
    from relight_engine.adapters.protocols import NeuralRelightPrediction

    h, w = 32, 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.full((h, w, 3), 0.45, dtype=np.float32)
    # Sharp high-frequency horizontal detail across the image
    rgb += 0.08 * np.sin(xx * 1.5)[..., None]
    rgb = np.clip(rgb, 0.0, 1.0)
    alpha = np.ones((h, w), dtype=np.float32)
    alpha[:, :4] = 0.0
    protected = np.zeros((h, w), dtype=np.float32)
    protected[:10, :] = 1.0
    lightable = 1.0 - protected
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=lightable,
        protected_mask=protected,
        alpha_mask=alpha,
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig = LightingRig(ambient_strength=0.2, lights=(
        Light("tgt", "directional", strength=0.9, direction=(0.3, 0.2, -0.9), casts_shadows=False),
    ))

    class SmoothNeuralRelighter:
        def render(self, sc, target_d, target_s, target_rig, settings):
            # Smooth brightened relight (no high-freq stripes, tries to overwrite protected/alpha pixels)
            return NeuralRelightPrediction(rgb_srgb=np.full_like(sc.rgb_srgb, 0.72))

        def runtime_snapshot(self):
            return {"device": "cpu"}

        def close(self):
            return {"released": False}

    with pytest.raises(ValueError, match="NeuralRelighter"):
        relight(scene, camera, rig, rig, EngineSettings(renderer_mode="hybrid_neural"))

    res = relight(
        scene, camera, rig, rig,
        EngineSettings(renderer_mode="hybrid_neural", base_blur_sigma=4.0, detail_lock_strength=1.0),
        neural_relighter=SmoothNeuralRelighter(),
    )
    # Protected top rows and transparent left columns stay locked to source
    assert np.allclose(res.rgb_srgb[:10, :, :], rgb[:10, :, :], atol=1e-5)
    assert np.allclose(res.rgb_srgb[:, :4, :], rgb[:, :4, :], atol=1e-5)
    # Active region takes neural brightness (~0.72) AND retains high-freq stripes from source
    active_patch = res.rgb_srgb[14:28, 10:40, 0]
    assert float(np.mean(active_patch)) > 0.62
    assert float(np.std(active_patch)) > 0.03
    assert res.metrics["renderer_mode"] == "hybrid_neural"


def test_hybrid_neural_rejects_bad_shape_nonfinite_and_out_of_range():
    from relight_engine.adapters.protocols import NeuralRelightPrediction

    scene, camera = make_plane()
    rig = LightingRig(ambient_strength=0.2, lights=(
        Light("tgt", "directional", strength=0.9, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    settings = EngineSettings(renderer_mode="hybrid_neural")

    class _BadRelighter:
        def __init__(self, arr):
            self.arr = arr

        def render(self, *_args):
            return NeuralRelightPrediction(rgb_srgb=self.arr)

        def runtime_snapshot(self):
            return {}

        def close(self):
            return {}

    with pytest.raises(ValueError, match="shape"):
        relight(scene, camera, rig, rig, settings, neural_relighter=_BadRelighter(np.zeros((2, 2, 3), np.float32)))
    bad_nan = np.full_like(scene.rgb_srgb, np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        relight(scene, camera, rig, rig, settings, neural_relighter=_BadRelighter(bad_nan))
    bad_range = np.full_like(scene.rgb_srgb, 1.5)
    with pytest.raises(ValueError, match="out-of-\\[0,1\\]"):
        relight(scene, camera, rig, rig, settings, neural_relighter=_BadRelighter(bad_range))


def test_auto_mode_defaults_to_source_preserving_ratio_even_when_albedo_present():
    h, w = 24, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.clip(0.45 + 0.15 * np.sin(xx * 0.4) * np.cos(yy * 0.3), 0.0, 1.0)[..., None]
    rgb = np.repeat(rgb, 3, axis=-1).astype(np.float32)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        # Intentionally flat blurry albedo — auto mode must keep original rgb identity
        albedo=np.full((h, w, 3), 0.5, dtype=np.float32),
        source_shading=np.ones((h, w), dtype=np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig = LightingRig(ambient_strength=0.2, lights=(
        Light("key", "directional", strength=0.8, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    res = relight(scene, camera, rig, rig, EngineSettings(renderer_mode="auto"))
    assert res.metrics["renderer_mode"] == "ratio"
    assert np.allclose(res.rgb_srgb, rgb, atol=1e-4)


def test_preserve_source_lighting_adds_colored_point_light_leaving_shadow_unchanged():
    h, w = 48, 64
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    depth = np.full((h, w), 4.0, dtype=np.float32)
    # Near blocker column on the right side casting shadow onto far-right edge from center-left bulb
    depth[10:38, 34:40] = 2.0
    rgb = np.full((h, w, 3), 0.50, dtype=np.float32)
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
    source_rig = LightingRig(ambient_strength=0.25, lights=(
        Light("src", "directional", strength=0.8, direction=(-0.5, 0.5, -0.5), casts_shadows=False),
    ))
    # Target rig specifies ONLY the added vivid red point light (no need to repeat source_rig)
    red_only_rig = LightingRig(ambient_strength=0.0, lights=(
        Light("red_bulb", "point", position=(-0.4, 0.0, 1.5), color=(1.0, 0.04, 0.04), strength=5.0),
    ))
    res = relight(
        scene, camera, source_rig, red_only_rig,
        EngineSettings(renderer_mode="ratio", preserve_source_lighting=True),
    )
    lit_pixel = res.rgb_srgb[24, 16]
    shadowed_pixel = res.rgb_srgb[24, 54]
    # Lit area receives vivid red glow while occluded area preserves original 0.50 gray photo
    assert float(lit_pixel[0] - lit_pixel[2]) > 0.20
    assert np.allclose(shadowed_pixel, 0.50, atol=0.04)


def test_active_mask_applies_full_relight_for_valid_confidence_prior():
    h, w = 24, 32
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene_075 = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.45, dtype=np.float32),
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.full((h, w), 0.75, dtype=np.float32),
    )
    scene_100 = SceneBuffers(
        rgb_srgb=scene_075.rgb_srgb,
        depth=scene_075.depth,
        depth_confidence=scene_075.depth_confidence,
        lightable_mask=scene_075.lightable_mask,
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    src = LightingRig(ambient_strength=0.25, lights=(
        Light("s", "directional", direction=(0.0, 0.0, -1.0), strength=0.8, casts_shadows=False),
    ))
    tgt = LightingRig(ambient_strength=0.10, lights=(
        Light("t", "directional", direction=(0.0, 0.0, -1.0), strength=1.5, color=(1.0, 0.2, 0.2), casts_shadows=False),
    ))
    r75 = relight(scene_075, camera, src, tgt, EngineSettings(renderer_mode="ratio"))
    r100 = relight(scene_100, camera, src, tgt, EngineSettings(renderer_mode="ratio"))
    assert np.allclose(r75.rgb_srgb, r100.rgb_srgb, atol=1e-4)


def test_lighting_smooth_suppresses_normal_noise_without_blurring_source_texture():
    rng = np.random.default_rng(7)
    h, w = 32, 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Crisp 2px periodic source texture must stay sharp
    tex = np.where(((xx.astype(int) + yy.astype(int)) % 2) == 0, 0.60, 0.35).astype(np.float32)
    rgb = np.repeat(tex[..., None], 3, axis=-1)
    # Noisy surface normals on left plane; sharp depth step at w//2 must not bleed light across
    depth = np.full((h, w), 2.0, dtype=np.float32)
    depth[:, w // 2 :] = 6.0
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 0] = rng.normal(0.0, 0.35, size=(h, w)).astype(np.float32)
    normals[..., 2] = -1.0
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    scene = SceneBuffers(
        rgb_srgb=rgb, depth=depth, depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32), depth_kind="relative",
        normals=normals, geometry_confidence=np.ones((h, w), np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    src = LightingRig(ambient_strength=0.3, lights=(
        Light("s", "directional", direction=(0.0, 0.0, -1.0), strength=0.7, casts_shadows=False),
    ))
    tgt = LightingRig(ambient_strength=0.1, lights=(
        Light("t", "directional", direction=(0.8, 0.0, -0.6), strength=1.4, color=(1.0, 0.2, 0.2), casts_shadows=False),
    ))
    raw_res = relight(scene, camera, src, tgt, EngineSettings(renderer_mode="ratio", lighting_smooth_sigma=0.0))
    smooth_res = relight(scene, camera, src, tgt, EngineSettings(renderer_mode="ratio", lighting_smooth_sigma=1.5))
    # Depth-guided lighting smoothing cuts noisy normal variation while preserving checkerboard contrast
    assert np.std(smooth_res.diffuse_ratio[:, 4 : w // 2 - 4]) < np.std(raw_res.diffuse_ratio[:, 4 : w // 2 - 4]) * 0.5
    checker_even = smooth_res.rgb_srgb[4:20:2, 4:18:2, 0].mean()
    checker_odd = smooth_res.rgb_srgb[4:20:2, 5:18:2, 0].mean()
    assert abs(checker_even - checker_odd) > 0.12


def test_shadow_max_thickness_prevents_far_background_streak_while_keeping_contact_shadow():
    h, w = 48, 64
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    depth = np.full((h, w), 4.8, dtype=np.float32)
    # Near small occluder far in front of the distant background
    depth[18:30, 20:24] = 1.4
    # Nearby surface right next to a second occluder step
    depth[:, 44:] = 1.8
    depth[18:30, 40:44] = 1.4
    points = backproject_depth(depth, camera)
    normals = np.zeros_like(points)
    normals[..., 2] = -1.0
    to_light = np.zeros_like(points)
    to_light[:] = np.array([-1.0, 0.0, -0.45], dtype=np.float32)

    vis_unbounded = ray_visibility(
        points, normals, depth, camera, to_light, np.full((h, w), 6.0, np.float32),
        "relative", EngineSettings(shadow_steps=20, shadow_max_thickness=None),
    )
    vis_bounded = ray_visibility(
        points, normals, depth, camera, to_light, np.full((h, w), 6.0, np.float32),
        "relative", EngineSettings(shadow_steps=20, shadow_max_thickness=0.7),
    )
    # Unbounded hits the far background across a 3.4-unit chasm; bounded does not streak the distant plane
    assert float(vis_unbounded[24, 28]) < 0.5
    assert float(vis_bounded[24, 28]) > 0.95
    # Contact shadow onto nearby 0.4-unit step remains intact
    assert float(vis_bounded[24, 46]) < 0.5


def test_wrapped_diffuse_and_hemispheric_ambient_smooth_terminator():
    h, w = 16, 16
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:, :8] = np.array([0.55, 0.2, -0.81], dtype=np.float32)  # just past 90-degree terminator (N.L ~ -0.15)
    normals[:, 8:] = np.array([-0.1, -0.95, -0.29], dtype=np.float32)  # faces down toward ground
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.45, dtype=np.float32),
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    pts = backproject_depth(scene.depth, camera)
    hard_rig = LightingRig(
        ambient_color=(0.2, 0.2, 0.8),
        ambient_ground_color=(0.9, 0.4, 0.2),
        ambient_strength=0.3,
        lights=(Light("sun", "directional", direction=(-0.9, 0.1, -0.4), color=(1.0, 0.7, 0.3), wrap=0.0, casts_shadows=False),),
    )
    wrap_rig = LightingRig(
        ambient_color=(0.2, 0.2, 0.8),
        ambient_ground_color=(0.9, 0.4, 0.2),
        ambient_strength=0.3,
        lights=(Light("sun", "directional", direction=(-0.9, 0.1, -0.4), color=(1.0, 0.7, 0.3), wrap=0.45, casts_shadows=False),),
    )
    d_hard, _, _ = evaluate_rig(scene, camera, pts, normals, hard_rig, EngineSettings())
    d_wrap, _, _ = evaluate_rig(scene, camera, pts, normals, wrap_rig, EngineSettings())
    # Wrap smoothly illuminates grazing faces past the hard 90-degree terminator
    assert float(d_wrap[8, 2, 0]) > float(d_hard[8, 2, 0]) + 0.10
    # Downward-facing normals pick up warm ground bounce more strongly than blue sky ambient
    assert float(d_hard[8, 12, 0]) > float(d_hard[8, 12, 2])


def test_source_shading_blend_and_aerial_perspective_stay_exact_in_foreground():
    from relight_engine.color import linear_to_srgb

    h, w = 32, 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = np.where(xx < w // 2, 1.0, 5.0).astype(np.float32)
    # Fine 2px texture across both foreground and background
    tex = np.where(((xx.astype(int) + yy.astype(int)) % 2) == 0, 0.55, 0.30).astype(np.float32)
    # Cool blue tint baked into background right half via source_shading
    albedo = np.repeat(tex[..., None], 3, axis=-1)
    src_sh = np.ones((h, w, 3), dtype=np.float32)
    src_sh[:, w // 2 :] = np.array([0.55, 0.75, 1.10], dtype=np.float32)
    rgb_srgb = linear_to_srgb(np.clip(albedo * src_sh, 0.0, 1.0))
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene = SceneBuffers(
        rgb_srgb=rgb_srgb, depth=depth, depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32), depth_kind="relative",
        normals=normals, geometry_confidence=np.ones((h, w), np.float32),
        source_shading=src_sh,
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    src_rig = LightingRig(ambient_strength=0.4, lights=(
        Light("src", "directional", direction=(0, 0, -1), strength=0.6, casts_shadows=False),
    ))
    tgt_rig = LightingRig(ambient_strength=0.35, ambient_color=(1.0, 0.78, 0.50), lights=(
        Light("warm", "directional", direction=(-0.3, 0.3, -0.9), color=(1.0, 0.72, 0.40), strength=0.9, casts_shadows=False),
    ))
    res = relight(
        scene, camera, src_rig, tgt_rig,
        EngineSettings(detail_power=1.0, source_shading_blend=0.8, atmosphere_density=0.25),
    )
    # Foreground remains sharp with checkerboard intact and warm relit
    fg_even = float(res.rgb_srgb[4:24:2, 4:18:2, 0].mean())
    fg_odd = float(res.rgb_srgb[4:24:2, 5:18:2, 0].mean())
    assert abs(fg_even - fg_odd) > 0.12
    # Cool blue background haze neutralizes and warms up instead of staying colder than red
    bg_mean = res.rgb_srgb[:, w // 2 + 4 : -4].mean(axis=(0, 1))
    assert float(bg_mean[0]) > float(bg_mean[2])


def test_normal_smooth_sigma_suppresses_texture_crease_across_flat_depth():
    h, w = 24, 32
    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[:, w // 2 :] = 1.5  # crisp depth boundary between two planes
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    # Fake 2px-wide normal glitch on flat left depth plane; distinct tilted right plane
    normals[10:14, 6:8] = np.array([0.85, 0.0, -0.52], dtype=np.float32)
    normals[:, w // 2 :] = np.array([-0.6, 0.0, -0.8], dtype=np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.5, dtype=np.float32),
        depth=depth, depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32), depth_kind="relative",
        normals=normals, geometry_confidence=np.ones((h, w), np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig = LightingRig(ambient_strength=0.1, lights=(
        Light("key", "directional", direction=(-0.7, 0.2, -0.7), strength=1.0, casts_shadows=False),
    ))
    res = relight(scene, camera, rig, rig, EngineSettings(normal_smooth_sigma=2.5))
    # Glitch smoothed away on flat left plane while tilted right plane normal remains distinct
    assert abs(float(res.normal_map[12, 7, 0])) < 0.25
    assert float(res.normal_map[12, w // 2 + 4, 0]) < -0.50
