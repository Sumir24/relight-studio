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


def test_effective_material_maps_broadcasts_specular_color_consistently():
    from relight_engine.lighting import _effective_material_maps
    scene, _ = make_plane()
    # 2D monochrome specular color
    scene.specular_color = np.full(scene.shape, 0.45, dtype=np.float32)
    *_, spec_col_resolved, _, _, _ = _effective_material_maps(scene)
    assert spec_col_resolved.shape == (*scene.shape, 3)
    assert np.allclose(spec_col_resolved[..., 0], 0.45)
    assert np.allclose(spec_col_resolved[..., 1], 0.45)
    assert np.allclose(spec_col_resolved[..., 2], 0.45)


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


def test_auto_mode_selects_intrinsic_when_albedo_available():
    from relight_engine.color import linear_to_srgb

    h, w = 24, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    albedo_lin = np.full((h, w, 3), 0.50, dtype=np.float32)
    source_sh = np.full((h, w), 0.35, dtype=np.float32)
    rgb = linear_to_srgb(albedo_lin * source_sh[..., None])
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
        albedo=linear_to_srgb(albedo_lin),
        source_shading=source_sh,
        non_diffuse_residual=np.full((h, w, 3), 0.05, dtype=np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig = LightingRig(ambient_strength=0.2, lights=(
        Light("key", "directional", strength=0.8, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    res_auto = relight(scene, camera, rig, rig, EngineSettings(renderer_mode="auto", base_blur_sigma=4.0))
    res_intr = relight(scene, camera, rig, rig, EngineSettings(renderer_mode="intrinsic", base_blur_sigma=4.0))
    assert res_auto.metrics["renderer_mode"] == "intrinsic"
    assert np.allclose(res_auto.rgb_srgb, res_intr.rgb_srgb, atol=0.0, rtol=0.0)


def test_auto_mode_falls_back_to_ratio_when_albedo_missing():
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
        albedo=None,
        source_shading=np.ones((h, w), dtype=np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig = LightingRig(ambient_strength=0.2, lights=(
        Light("key", "directional", strength=0.8, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    res_auto = relight(scene, camera, rig, rig, EngineSettings(renderer_mode="auto"))
    res_ratio = relight(scene, camera, rig, rig, EngineSettings(renderer_mode="ratio"))
    assert res_auto.metrics["renderer_mode"] == "ratio"
    assert np.allclose(res_auto.rgb_srgb, res_ratio.rgb_srgb, atol=0.0, rtol=0.0)
    assert np.allclose(res_auto.rgb_srgb, rgb, atol=1e-4)


def test_explicit_ratio_mode_overrides_available_intrinsic_buffers():
    from relight_engine.color import linear_to_srgb

    h, w = 24, 32
    albedo_lin = np.full((h, w, 3), 0.50, dtype=np.float32)
    source_sh = np.full((h, w), 0.35, dtype=np.float32)
    rgb = linear_to_srgb(albedo_lin * source_sh[..., None])
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
        albedo=linear_to_srgb(albedo_lin),
        source_shading=source_sh,
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig = LightingRig(ambient_strength=0.2, lights=(
        Light("key", "directional", strength=0.8, direction=(0.0, 0.0, -1.0), casts_shadows=False),
    ))
    res_ratio = relight(scene, camera, rig, rig, EngineSettings(renderer_mode="ratio", base_blur_sigma=4.0))
    assert res_ratio.metrics["renderer_mode"] == "ratio"
    assert np.allclose(res_ratio.rgb_srgb, rgb, atol=1e-4)


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


def test_indirect_bounce_bleeds_color_onto_facing_neighbor():
    h, w = 32, 48
    rgb = np.full((h, w, 3), 0.45, dtype=np.float32)
    # Left half is vivid red emitter surface, right half is neutral grey wall tilted toward left
    rgb[:, : w // 2] = np.array([0.88, 0.08, 0.08], dtype=np.float32)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:, : w // 2] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    normals[:, w // 2 :] = np.array([-0.75, 0.0, -0.66], dtype=np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    scene = SceneBuffers(
        rgb_srgb=rgb, depth=depth, depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32), depth_kind="relative",
        normals=normals, geometry_confidence=np.ones((h, w), np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    src = LightingRig(ambient_strength=0.4, lights=())
    tgt = LightingRig(ambient_strength=0.1, lights=(
        Light("key", "directional", direction=(0.6, 0.0, -0.8), strength=1.5, casts_shadows=False),
    ))
    no_gi = relight(scene, camera, src, tgt, EngineSettings(indirect_bounce_strength=0.0))
    with_gi = relight(scene, camera, src, tgt, EngineSettings(indirect_bounce_strength=0.65, indirect_bounce_radius=12.0))
    # Right side near the boundary catches red indirect color bleed from the red left half
    diff_near = with_gi.rgb_srgb[16, w // 2 + 4] - no_gi.rgb_srgb[16, w // 2 + 4]
    assert float(diff_near[0]) > float(diff_near[2]) + 0.02


def test_clearcoat_and_subsurface_respond_to_multi_light():
    h, w = 16, 16
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:, :8] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    normals[:, 8:] = np.array([0.92, 0.0, -0.39], dtype=np.float32)  # grazing ~90-deg terminator to left light
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    scene_base = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.45, np.float32), depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.ones((h, w), np.float32), lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative", normals=normals, geometry_confidence=np.ones((h, w), np.float32),
        roughness=np.full((h, w), 0.85, np.float32),
    )
    scene_layered = SceneBuffers(
        rgb_srgb=scene_base.rgb_srgb, depth=scene_base.depth,
        depth_confidence=scene_base.depth_confidence, lightable_mask=scene_base.lightable_mask,
        depth_kind="relative", normals=normals, geometry_confidence=scene_base.geometry_confidence,
        roughness=scene_base.roughness,
        clearcoat=np.full((h, w), 0.9, np.float32),
        clearcoat_roughness=np.full((h, w), 0.08, np.float32),
        subsurface=np.full((h, w), 0.8, np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    pts = backproject_depth(scene_base.depth, camera)
    rig = LightingRig(ambient_strength=0.05, lights=(
        Light("front_spec", "directional", direction=(0.0, 0.0, -1.0), strength=1.2, casts_shadows=False),
        Light("front_left", "directional", direction=(-0.4, 0.0, -0.9), strength=1.2, casts_shadows=False),
    ))
    d_base, s_base, _ = evaluate_rig(scene_base, camera, pts, normals, rig, EngineSettings())
    d_lay, s_lay, _ = evaluate_rig(scene_layered, camera, pts, normals, rig, EngineSettings())
    # Clear coat adds crisp highlight specular over rough base
    assert float(np.max(s_lay[:, :8])) > float(np.max(s_base[:, :8])) + 0.50
    # Subsurface adds warm red scatter along grazing terminator without washing out front-lit skin
    assert float(d_lay[8, 12, 0] - d_base[8, 12, 0]) > float(d_lay[8, 12, 2] - d_base[8, 12, 2]) + 0.02
    assert np.allclose(d_lay[8, 2], d_base[8, 2], atol=0.02)


def test_auto_sh_source_and_filmic_rolloff_and_edge_depth_refine():
    from relight_engine.color import linear_to_srgb

    h, w = 24, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Two-light baked source shading (warm left + cool top) baked across curved cylinder normals
    angle = (xx / (w - 1) - 0.5) * 2.2
    normals = np.stack([np.sin(angle), np.zeros_like(angle), -np.cos(angle)], axis=-1).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    # Approximate true multi-light shading on the cylinder
    l1 = np.clip(np.sum(normals * np.array([-0.8, 0.0, -0.6]), axis=-1), 0, 1)[..., None] * np.array([1.2, 0.8, 0.4])
    l2 = np.clip(np.sum(normals * np.array([0.8, 0.2, -0.57]), axis=-1), 0, 1)[..., None] * np.array([0.4, 0.7, 1.3])
    true_sh = (0.25 + l1 + l2).astype(np.float32)

    # Sharp foreground edge with blurry relative depth transition
    rgb_lin = np.clip(0.35 * true_sh, 0.0, 1.0)
    rgb_lin[:, : w // 2] *= 1.4
    depth = np.clip(2.0 + 2.0 * (1.0 / (1.0 + np.exp(-(xx - w // 2) * 0.5))), 1.0, 5.0).astype(np.float32)
    scene = SceneBuffers(
        rgb_srgb=linear_to_srgb(rgb_lin), depth=depth, depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32), depth_kind="relative",
        normals=normals, geometry_confidence=np.ones((h, w), np.float32),
        source_shading=true_sh,
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    src = LightingRig(ambient_strength=0.3, lights=(Light("guess", "directional", direction=(0, 0, -1), casts_shadows=False),))
    # Bright overlapping dual-light rig to exercise filmic highlight rolloff
    tgt = LightingRig(ambient_strength=0.2, lights=(
        Light("warm", "directional", direction=(-0.6, 0.3, -0.7), color=(1.0, 0.7, 0.3), strength=1.8, radius=0.15, casts_shadows=False),
        Light("blue", "directional", direction=(0.6, 0.3, -0.7), color=(0.3, 0.6, 1.0), strength=1.8, radius=0.15, casts_shadows=False),
    ))
    res = relight(
        scene, camera, src, tgt,
        EngineSettings(
            detail_power=1.0,
            auto_sh_source=True,
            filmic_highlight_rolloff=True,
            refine_depth_edges=True,
        ),
    )
    assert np.isfinite(res.rgb_srgb).all()
    assert float(np.max(res.rgb_srgb)) <= 1.0
    assert float(np.mean(np.abs(res.rgb_srgb - scene.rgb_srgb))) > 0.02


def test_pcss_light_radius_softens_far_shadow_more_than_contact_shadow():
    h, w = 48, 64
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    depth = np.full((h, w), 4.5, dtype=np.float32)
    depth[18:30, 18:22] = 1.5
    points = backproject_depth(depth, camera)
    normals = np.zeros_like(points)
    normals[..., 2] = -1.0
    to_light = np.zeros_like(points)
    to_light[:] = np.array([-1.0, 0.0, -0.45], dtype=np.float32)
    max_dist = np.full((h, w), 6.0, np.float32)
    vis_hard = ray_visibility(points, normals, depth, camera, to_light, max_dist, "relative", EngineSettings(shadow_steps=20), light_radius=0.0)
    vis_soft = ray_visibility(points, normals, depth, camera, to_light, max_dist, "relative", EngineSettings(shadow_steps=20), light_radius=0.35)
    # Soft radius light introduces smooth gradient differences across penumbra boundary
    assert float(np.max(np.abs(vis_soft - vis_hard))) > 0.05


def test_intrinsic_chroma_preservation_restores_fine_color_detail_without_source_casts_or_halos():
    from relight_engine.color import linear_to_srgb, luminance, srgb_to_linear

    h, w = 48, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0

    # 1. Fine high-frequency chromatic variation in source RGB (alternating warm/cool stripes)
    #    while predicted albedo is spatially smoothed neutral grey.
    albedo_lin = np.full((h, w, 3), 0.35, dtype=np.float32)
    stripe = np.sin(xx * 1.4)
    src_lin = albedo_lin.copy()
    src_lin[..., 0] += 0.10 * stripe
    src_lin[..., 2] -= 0.10 * stripe
    src_lin = np.clip(src_lin, 0.01, 0.99)
    scene_fine = SceneBuffers(
        rgb_srgb=linear_to_srgb(src_lin),
        depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), np.float32),
        albedo=linear_to_srgb(albedo_lin),
        source_shading=np.ones((h, w), np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig_dim = LightingRig(ambient_strength=0.15, lights=(
        Light("key", "directional", direction=(0.0, 0.0, -1.0), strength=0.35, casts_shadows=False),
    ))
    rig_bright = LightingRig(ambient_strength=0.25, lights=(
        Light("key", "directional", direction=(0.0, 0.0, -1.0), strength=0.95, casts_shadows=False),
    ))
    res_fine = relight(scene_fine, camera, rig_bright, rig_bright, EngineSettings(renderer_mode="intrinsic", base_blur_sigma=8.0))
    res_dim = relight(scene_fine, camera, rig_bright, rig_dim, EngineSettings(renderer_mode="intrinsic", base_blur_sigma=8.0))

    # Fine R-B chromatic oscillation must survive instead of collapsing to neutral grey
    rb_diff = res_fine.rgb_srgb[12:36, 12:52, 0] - res_fine.rgb_srgb[12:36, 12:52, 2]
    assert float(np.std(rb_diff)) > 0.04
    # Relighting still changes luminance according to requested target rig
    assert float(np.mean(luminance(srgb_to_linear(res_fine.rgb_srgb)))) > float(np.mean(luminance(srgb_to_linear(res_dim.rgb_srgb)))) + 0.12

    # 2. Broad low-frequency source color cast gradient on smooth albedo must NOT be inherited
    grad = np.linspace(-0.08, 0.08, w, dtype=np.float32)[None, :]
    cast_lin = albedo_lin.copy()
    cast_lin[..., 0] += grad
    cast_lin[..., 2] -= grad
    scene_cast = SceneBuffers(
        rgb_srgb=linear_to_srgb(np.clip(cast_lin, 0.01, 0.99)),
        depth=scene_fine.depth,
        depth_confidence=scene_fine.depth_confidence,
        lightable_mask=scene_fine.lightable_mask,
        depth_kind="relative",
        normals=normals,
        geometry_confidence=scene_fine.geometry_confidence,
        albedo=linear_to_srgb(albedo_lin),
        source_shading=np.ones((h, w), np.float32),
    )
    res_cast = relight(scene_cast, camera, rig_bright, rig_bright, EngineSettings(renderer_mode="intrinsic", base_blur_sigma=8.0))
    left_rb = float(np.mean(res_cast.rgb_srgb[8:40, 4:12, 0] - res_cast.rgb_srgb[8:40, 4:12, 2]))
    right_rb = float(np.mean(res_cast.rgb_srgb[8:40, w - 12 : w - 4, 0] - res_cast.rgb_srgb[8:40, w - 12 : w - 4, 2]))
    assert abs(right_rb - left_rb) < 0.04

    # 3. Sharp step edge between vivid red object and neutral grey background must not produce color halos
    step_lin = np.full((h, w, 3), 0.35, dtype=np.float32)
    step_lin[:, : w // 2] = np.array([0.55, 0.20, 0.20], dtype=np.float32)
    protected = np.zeros((h, w), dtype=np.float32)
    protected[:6, :] = 1.0
    scene_step = SceneBuffers(
        rgb_srgb=linear_to_srgb(step_lin),
        depth=scene_fine.depth,
        depth_confidence=scene_fine.depth_confidence,
        lightable_mask=1.0 - protected,
        protected_mask=protected,
        depth_kind="relative",
        normals=normals,
        geometry_confidence=scene_fine.geometry_confidence,
        albedo=linear_to_srgb(step_lin),
        source_shading=np.ones((h, w), np.float32),
    )
    res_step = relight(scene_step, camera, rig_bright, rig_bright, EngineSettings(renderer_mode="intrinsic", base_blur_sigma=8.0))
    # Protected rows remain exact
    assert np.allclose(res_step.rgb_srgb[:6, :], scene_step.rgb_srgb[:6, :], atol=1e-6)
    # Neutral right side near the boundary (w//2 + 2) stays neutral without red halo bleed
    near_edge_rb = float(np.mean(res_step.rgb_srgb[12:36, w // 2 + 1 : w // 2 + 5, 0] - res_step.rgb_srgb[12:36, w // 2 + 1 : w // 2 + 5, 2]))
    assert abs(near_edge_rb) < 0.015

    # 4. Exact no-op when albedo matches src / source_shading and target_rig == unit shading
    unit_rig = LightingRig(ambient_strength=1.0, lights=())
    res_noop = relight(scene_step, camera, unit_rig, unit_rig, EngineSettings(renderer_mode="intrinsic", base_blur_sigma=8.0))
    assert np.allclose(res_noop.rgb_srgb, scene_step.rgb_srgb, atol=1e-4)


def test_optional_f3_detail_mode_and_cache_invalidation():
    from relight_engine.color import linear_to_srgb, luminance, srgb_to_linear

    h, w = 48, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0

    # Texture combining medium-scale structure (~10px) and fine detail (~3px) smoothed out of albedo
    texture = 0.06 * np.sin(xx * 0.6) + 0.05 * np.sin(xx * 1.8) * np.cos(yy * 1.2)
    src_lin = np.clip(0.38 + texture, 0.02, 0.98)[..., None].repeat(3, axis=-1).astype(np.float32)
    albedo_lin = np.full((h, w, 3), 0.38, dtype=np.float32)
    protected = np.zeros((h, w), dtype=np.float32)
    protected[:6, :] = 1.0

    scene = SceneBuffers(
        rgb_srgb=linear_to_srgb(src_lin),
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=1.0 - protected,
        protected_mask=protected,
        alpha_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        albedo=linear_to_srgb(albedo_lin),
        source_shading=np.ones((h, w), dtype=np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig_a = LightingRig(ambient_strength=0.2, lights=(
        Light("key_r", "directional", direction=(0.6, 0.2, -0.7), strength=0.9, casts_shadows=False),
    ))
    rig_b = LightingRig(ambient_strength=0.2, lights=(
        Light("key_l", "directional", direction=(-0.6, 0.2, -0.7), strength=1.2, casts_shadows=False),
    ))

    # Default stays 'baseline'
    assert EngineSettings().detail_preservation_mode == "baseline"
    res_def = relight(scene, camera, rig_a, rig_a, EngineSettings(renderer_mode="intrinsic"))
    res_base = relight(scene, camera, rig_a, rig_a, EngineSettings(renderer_mode="intrinsic", detail_preservation_mode="baseline"))
    assert np.allclose(res_def.rgb_srgb, res_base.rgb_srgb, atol=0.0, rtol=0.0)

    # F3 recovers more structured luminance variation than baseline while keeping protected pixels exact
    res_f3_a = relight(scene, camera, rig_a, rig_a, EngineSettings(renderer_mode="intrinsic", detail_preservation_mode="f3"))
    assert np.allclose(res_f3_a.rgb_srgb[:6, :], scene.rgb_srgb[:6, :], atol=1e-6)
    std_base = float(np.std(res_base.rgb_srgb[12:40, 12:52]))
    std_f3 = float(np.std(res_f3_a.rgb_srgb[12:40, 12:52]))
    assert std_f3 >= std_base

    # Cache hit on different target rig keeps exact lighting response & invalidates on buffer mutation
    res_f3_b = relight(scene, camera, rig_a, rig_b, EngineSettings(renderer_mode="intrinsic", detail_preservation_mode="f3"))
    assert float(np.mean(luminance(srgb_to_linear(res_f3_b.rgb_srgb[12:40, :])))) != pytest.approx(
        float(np.mean(luminance(srgb_to_linear(res_f3_a.rgb_srgb[12:40, :])))), abs=1e-3
    )

    scene.albedo[:] = linear_to_srgb(np.full((h, w, 3), 0.62, dtype=np.float32))
    res_mutated = relight(scene, camera, rig_a, rig_a, EngineSettings(renderer_mode="intrinsic", detail_preservation_mode="f3"))
    assert float(np.mean(res_mutated.rgb_srgb[12:40, 12:52])) > float(np.mean(res_f3_a.rgb_srgb[12:40, 12:52])) + 0.05

    # Ratio renderer remains unaffected regardless of detail_preservation_mode
    res_ratio_base = relight(scene, camera, rig_a, rig_b, EngineSettings(renderer_mode="ratio", detail_preservation_mode="baseline"))
    res_ratio_f3 = relight(scene, camera, rig_a, rig_b, EngineSettings(renderer_mode="ratio", detail_preservation_mode="f3"))
    assert np.allclose(res_ratio_base.rgb_srgb, res_ratio_f3.rgb_srgb, atol=0.0, rtol=0.0)


def test_adaptive_bias_suppresses_grazing_acne_while_preserving_contact_shadows():
    h, w = 96, 96
    camera = CameraModel(120.0, 120.0, w / 2, h / 2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx_px = xx - 48.0
    dy_px = yy - 48.0
    r2 = dx_px * dx_px + dy_px * dy_px
    r_px = 20.0
    inside = r2 < r_px * r_px
    bg_z = np.full((h, w), 3.2, dtype=np.float32)
    dome = np.where(inside, np.sqrt(np.maximum(r_px * r_px - r2, 0.0)) * (3.2 / 120.0), 0.0)
    depth = (bg_z - dome).astype(np.float32)
    points = backproject_depth(depth, camera)
    normals, _ = estimate_normals(points, depth, np.ones((h, w), dtype=np.float32))

    # Grazing light from right (+X)
    l_dir = np.array([0.85, 0.20, -0.49], dtype=np.float32)
    l_dir /= np.linalg.norm(l_dir)
    to_light = np.broadcast_to(l_dir, (h, w, 3))
    ndotl = np.sum(normals * l_dir, axis=-1)

    vis_adaptive = ray_visibility(
        points, normals, depth, camera, to_light, np.full((h, w), 3.0, np.float32),
        "relative", EngineSettings(shadow_steps=24, shadow_adaptive_sampling=True),
        depth_confidence=np.full((h, w), 0.45, dtype=np.float32),
    )
    vis_dense = ray_visibility(
        points, normals, depth, camera, to_light, np.full((h, w), 3.0, np.float32),
        "relative", EngineSettings(shadow_steps=24, shadow_adaptive_sampling=False),
        depth_confidence=np.full((h, w), 0.45, dtype=np.float32),
    )

    grazing_lit = inside & (dx_px > 0) & (ndotl > 0.05) & (ndotl < 0.30)
    contact_shadow = (~inside) & (np.abs(dy_px) < 8.0) & (dx_px < -r_px - 1.0) & (dx_px > -r_px - 6.0)
    assert float(np.mean(vis_adaptive[grazing_lit])) > 0.90
    assert float(np.mean(vis_adaptive[contact_shadow])) < 0.45
    assert float(np.mean(np.abs(vis_adaptive - vis_dense))) < 0.02


def test_preview_and_final_quality_modes_and_lighting_scale_metrics():
    h, w = 80, 96
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = np.where(xx < 40, 1.8, 3.4).astype(np.float32)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    protected = np.zeros((h, w), dtype=np.float32)
    protected[:8, :] = 1.0
    alpha = np.full((h, w), 0.75, dtype=np.float32)
    scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.42, dtype=np.float32),
        depth=depth,
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=1.0 - protected,
        protected_mask=protected,
        alpha_mask=alpha,
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        albedo=np.full((h, w, 3), 0.50, dtype=np.float32),
        source_shading=np.ones((h, w), dtype=np.float32),
    )
    camera = CameraModel(80.0, 80.0, w / 2, h / 2)
    src_rig = LightingRig(ambient_strength=0.4, lights=(Light("s", "directional", direction=(0, 0, -1), casts_shadows=False),))
    tgt_rig = LightingRig(ambient_strength=0.15, lights=(Light("k", "directional", direction=(-0.7, 0.1, -0.7), strength=1.2, casts_shadows=True),))

    # Preview with depth-guided upsampling preserves full output resolution & protected pixels
    res_prev_up = relight(
        scene, camera, src_rig, tgt_rig,
        EngineSettings(quality_mode="preview", preview_max_side=48, preview_upsample=True),
    )
    assert res_prev_up.rgb_srgb.shape == (h, w, 3)
    assert res_prev_up.metrics["quality_mode"] == "preview"
    assert res_prev_up.metrics["preview_downscaled"] is True
    assert res_prev_up.metrics["preview_upsampled"] is True
    assert res_prev_up.metrics["shadow_steps_effective"] == 12
    assert np.allclose(res_prev_up.rgb_srgb[:8, :], scene.rgb_srgb[:8, :], atol=1e-6)
    assert res_prev_up.metrics["lighting_scale"]["calibrated_metric_scale"] is False

    # Native downscaled preview without upsampling
    res_prev_lo = relight(
        scene, camera, src_rig, tgt_rig,
        EngineSettings(quality_mode="preview", preview_max_side=48, preview_upsample=False),
    )
    assert max(res_prev_lo.rgb_srgb.shape[:2]) == 48
    assert res_prev_lo.metrics["preview_downscaled"] is True
    assert res_prev_lo.metrics["preview_upsampled"] is False

    # Final quality mode runs at full res with >= 48 shadow steps
    res_final = relight(
        scene, camera, src_rig, tgt_rig,
        EngineSettings(quality_mode="final"),
    )
    assert res_final.rgb_srgb.shape == (h, w, 3)
    assert res_final.metrics["quality_mode"] == "final"
    assert res_final.metrics["shadow_steps_effective"] >= 48
    assert float(np.mean(np.abs(res_prev_up.rgb_srgb - res_final.rgb_srgb))) < 0.04

    # RelightSession caches scene geometry/priors and reproduces relight() cleanly
    from relight_engine import prepare_relight_session
    session = prepare_relight_session(
        scene, camera, src_rig,
        EngineSettings(quality_mode="standard"), preview_max_side=48,
    )
    sess_prev_up = session.render(tgt_rig, EngineSettings(quality_mode="preview", preview_max_side=48, preview_upsample=True))
    sess_prev_lo = session.render(tgt_rig, EngineSettings(quality_mode="preview", preview_max_side=48, preview_upsample=False))
    sess_final = session.render(tgt_rig, EngineSettings(quality_mode="final"))
    assert sess_prev_up.metrics["session_cached"] is True
    assert np.allclose(sess_prev_up.rgb_srgb, res_prev_up.rgb_srgb, atol=1e-5)
    assert np.allclose(sess_prev_lo.rgb_srgb, res_prev_lo.rgb_srgb, atol=1e-5)
    assert np.allclose(sess_final.rgb_srgb, res_final.rgb_srgb, atol=1e-5)


def test_phase6_intrinsic_residual_separation_suppresses_shadow_glare_and_tints_with_target():
    from relight_engine.color import linear_to_srgb
    from relight_engine import prepare_relight_session

    h, w = 32, 48
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    depth = np.full((h, w), 3.0, dtype=np.float32)
    # Near blocker column on center-left so right half falls into cast shadow from left directional light
    depth[:, 18:22] = 1.2
    albedo_lin = np.full((h, w, 3), 0.35, dtype=np.float32)
    residual = np.full((h, w, 3), 0.80, dtype=np.float32)  # strong baked white source specular glare
    source_sh = np.full((h, w), 1.0, dtype=np.float32)

    scene = SceneBuffers(
        rgb_srgb=linear_to_srgb(np.clip(albedo_lin * source_sh[..., None] + residual * 0.2, 0.0, 1.0)),
        depth=depth,
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        albedo=linear_to_srgb(albedo_lin),
        source_shading=source_sh,
        non_diffuse_residual=residual,
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    src_rig = LightingRig(ambient_strength=0.3, lights=(
        Light("src", "directional", direction=(0.0, 0.0, -1.0), strength=0.7, casts_shadows=False),
    ))
    red_rig = LightingRig(ambient_color=(0.08, 0.04, 0.04), ambient_strength=0.08, lights=(
        Light("red_left", "directional", direction=(-0.85, 0.0, -0.52), color=(1.0, 0.05, 0.08), strength=1.5, casts_shadows=True),
    ))
    res = relight(scene, camera, src_rig, red_rig, EngineSettings(renderer_mode="intrinsic"))
    session = prepare_relight_session(scene, camera, src_rig, EngineSettings(renderer_mode="intrinsic"))
    res_sess = session.render(red_rig)
    assert np.allclose(res.rgb_srgb, res_sess.rgb_srgb, atol=1e-5)

    # Lit left patch takes vivid red illumination; shadowed right patch stays dark without a 0.25x white glare floor
    lit_px = res.rgb_srgb[16, 8]
    shadow_px = res.rgb_srgb[16, 36]
    assert float(lit_px[0] - lit_px[2]) > 0.25
    assert float(np.mean(shadow_px)) < 0.16


def test_phase6_ratio_renderer_avoids_global_red_tint_and_suppresses_old_white_glare():
    h, w = 40, 64
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[:, 24:28] = 1.2  # casts shadow onto right half from left red light
    rgb = np.full((h, w, 3), 0.35, dtype=np.float32)
    # Baked white highlight blob inside the shadowed right zone
    rgb[12:28, 42:54] = 0.92
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
    src_rig = LightingRig(ambient_strength=0.6, lights=(
        Light("src", "directional", direction=(0.0, 0.0, -1.0), strength=0.6, casts_shadows=False),
    ))
    red_rig = LightingRig(ambient_color=(0.1, 0.02, 0.03), ambient_strength=0.10, lights=(
        Light("red_left", "directional", direction=(-0.85, 0.0, -0.52), color=(1.0, 0.05, 0.10), strength=2.4, casts_shadows=True),
    ))
    res = relight(scene, camera, src_rig, red_rig, EngineSettings(renderer_mode="ratio", base_blur_sigma=4.0))
    lit_px = res.rgb_srgb[20, 10]
    shadow_old_blob = res.rgb_srgb[20, 48]
    # Lit side glows red while the shadowed old white blob stays dark and neutral rather than turning into a red spotlight
    assert float(lit_px[0] - lit_px[2]) > 0.25
    assert float(shadow_old_blob[0]) < float(lit_px[0]) * 0.70
    assert float(shadow_old_blob[0] - shadow_old_blob[2]) < 0.03


def test_phase7_observability_reports_ratio_fallback_and_intrinsic_paths():
    scene, camera = make_plane()
    rig = LightingRig(ambient_strength=0.2, lights=(
        Light("key", "directional", direction=(0.0, 0.0, -1.0), strength=1.0, casts_shadows=False),
    ))
    res_fallback = relight(scene, camera, rig, rig, EngineSettings(renderer_mode="auto"))
    assert res_fallback.metrics["requested_renderer_mode"] == "auto"
    assert res_fallback.metrics["renderer_mode"] == "ratio"
    assert res_fallback.metrics["fallback_reason"] == "intrinsic_albedo_unavailable_using_ratio_fallback"
    assert res_fallback.metrics["scene_buffers_available"]["albedo"] is False
    assert res_fallback.metrics["residual_decomposition_active"] is False
    assert res_fallback.metrics["environment_lighting_active"] is False
    assert res_fallback.metrics["active_specular_path"] == "authored_material_ggx"

    scene_intr = SceneBuffers(
        rgb_srgb=scene.rgb_srgb, depth=scene.depth, depth_confidence=scene.depth_confidence,
        lightable_mask=scene.lightable_mask, depth_kind="relative",
        albedo=scene.rgb_srgb, source_shading=np.ones(scene.shape, np.float32),
        non_diffuse_residual=np.full((*scene.shape, 3), 0.08, np.float32),
    )
    res_intr = relight(scene_intr, camera, rig, rig, EngineSettings(renderer_mode="auto"))
    assert res_intr.metrics["renderer_mode"] == "intrinsic"
    assert res_intr.metrics["fallback_reason"] is None
    assert res_intr.metrics["residual_decomposition_active"] is True
    assert res_intr.metrics["active_specular_path"] == "conservative_prior_dielectric_ggx"


def test_phase7_environment_light_studio_probe_and_validation():
    from relight_engine import EnvironmentLight

    scene, camera = make_plane()
    scene.roughness[:] = 0.18
    scene.metalness[:] = 1.0
    scene.specular_color = np.full((*scene.shape, 3), 0.85, dtype=np.float32)
    pts = backproject_depth(scene.depth, camera)
    nrm = np.zeros_like(pts)
    nrm[..., 2] = -1.0

    env = EnvironmentLight(
        sky_color=(0.4, 0.7, 1.0),
        horizon_color=(0.8, 0.85, 0.9),
        ground_color=(0.2, 0.15, 0.1),
        strength=0.6,
        key_azimuth_deg=0.0,
        key_elevation_deg=0.0,
        key_color=(1.0, 0.85, 0.7),
        key_strength=1.2,
        key_sharpness=8.0,
    )
    rig_no_env = LightingRig(ambient_strength=0.05, lights=())
    rig_with_env = LightingRig(ambient_strength=0.05, lights=(), environment=env)

    d0, s0, _ = evaluate_rig(scene, camera, pts, nrm, rig_no_env, EngineSettings())
    d1, s1, _ = evaluate_rig(scene, camera, pts, nrm, rig_with_env, EngineSettings())
    assert float(np.mean(s1)) > float(np.mean(s0)) + 0.25
    res_env = relight(scene, camera, rig_no_env, rig_with_env)
    assert res_env.metrics["environment_lighting_active"] is True
    assert "+environment_ibl" in res_env.metrics["active_specular_path"]

    with pytest.raises(ValueError, match="key_sharpness"):
        EnvironmentLight(key_sharpness=0.0).validate()


def test_phase7_ratio_preserves_bright_diffuse_surfaces_without_darkening():
    h, w = 24, 32
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    # Uniform bright white wall / shirt albedo (0.78 sRGB ~ 0.57 linear) must stay bright under equal-strength directional relight
    rgb = np.full((h, w, 3), 0.78, dtype=np.float32)
    scene = SceneBuffers(
        rgb_srgb=rgb, depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.ones((h, w), np.float32), lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative", normals=normals, geometry_confidence=np.ones((h, w), np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    src = LightingRig(ambient_strength=0.3, lights=(Light("s", "directional", direction=(0, 0, -1), strength=0.7, casts_shadows=False),))
    tgt = LightingRig(ambient_strength=0.3, lights=(Light("t", "directional", direction=(-0.5, 0.2, -0.84), strength=0.83, casts_shadows=False),))
    res = relight(scene, camera, src, tgt, EngineSettings(renderer_mode="ratio"))
    assert abs(float(np.mean(res.rgb_srgb)) - 0.78) < 0.05


def test_phase7_intrinsic_residual_untinted_glare_and_low_conf_guard():
    from relight_engine.color import linear_to_srgb

    h, w = 32, 48
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    albedo_lin = np.full((h, w, 3), 0.25, dtype=np.float32)
    # Broad white source reflection in non_diffuse_residual
    residual = np.full((h, w, 3), 0.50, dtype=np.float32)
    scene = SceneBuffers(
        rgb_srgb=linear_to_srgb(np.clip(albedo_lin + residual * 0.5, 0.0, 1.0)),
        depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), np.float32),
        albedo=linear_to_srgb(albedo_lin),
        source_shading=np.ones((h, w), np.float32),
        non_diffuse_residual=residual,
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    src = LightingRig(ambient_strength=0.3, lights=(Light("s", "directional", direction=(0, 0, -1), strength=0.7, casts_shadows=False),))
    red = LightingRig(ambient_color=(0.1, 0.05, 0.05), ambient_strength=0.05, lights=(
        Light("r", "directional", direction=(0, 0, -1), color=(1.0, 0.02, 0.02), strength=1.0, casts_shadows=False),
    ))
    res_with_res = relight(scene, camera, src, red, EngineSettings(renderer_mode="intrinsic"))
    scene_no_res = SceneBuffers(
        rgb_srgb=scene.rgb_srgb, depth=scene.depth, depth_confidence=scene.depth_confidence,
        lightable_mask=scene.lightable_mask, depth_kind="relative", normals=normals,
        geometry_confidence=scene.geometry_confidence, albedo=scene.albedo,
        source_shading=scene.source_shading, non_diffuse_residual=None,
    )
    res_no_res = relight(scene_no_res, camera, src, red, EngineSettings(renderer_mode="intrinsic"))
    # Broad original reflection contribution stays untinted: blue/green difference from residual is nearly neutral, not bright red!
    res_delta = res_with_res.rgb_srgb - res_no_res.rgb_srgb
    assert float(np.mean(res_delta[..., 0] - res_delta[..., 2])) < 0.03



def test_phase6_metallic_sheen_and_finite_radius_specular_response():
    scene, camera = make_plane()
    scene.roughness[:] = 0.20
    scene.metalness[:] = 1.0
    scene.specular_color = np.full((*scene.shape, 3), 0.85, dtype=np.float32)
    pts = backproject_depth(scene.depth, camera)
    nrm = np.zeros_like(pts)
    nrm[..., 2] = -1.0

    rig_point_zero_r = LightingRig(ambient_strength=0.0, lights=(
        Light("pt", "point", position=(-0.4, 0.0, 1.5), color=(1.0, 0.1, 0.1), strength=3.0, radius=0.0, casts_shadows=False),
    ))
    rig_point_finite_r = LightingRig(ambient_strength=0.0, lights=(
        Light("pt", "point", position=(-0.4, 0.0, 1.5), color=(1.0, 0.1, 0.1), strength=3.0, radius=0.15, casts_shadows=False),
    ))
    _, spec_zero_r, _ = evaluate_rig(scene, camera, pts, nrm, rig_point_zero_r, EngineSettings())
    _, spec_finite_r, _ = evaluate_rig(scene, camera, pts, nrm, rig_point_finite_r, EngineSettings())
    # Finite radius broadens metallic specular highlights across neighboring off-axis normals
    assert float(np.mean(spec_finite_r[..., 0] > 0.05)) >= float(np.mean(spec_zero_r[..., 0] > 0.05))
    assert float(np.max(spec_finite_r[..., 0])) > float(np.max(spec_finite_r[..., 2])) * 4.0

