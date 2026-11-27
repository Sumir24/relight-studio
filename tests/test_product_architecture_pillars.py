from __future__ import annotations

from dataclasses import replace
import numpy as np

from relight_engine.calibration import calibrate_conductor_from_appearance, compute_skin_likelihood_map
from relight_engine.contracts import CameraModel, EngineSettings, Light, LightingRig, SceneBuffers, SkinFidelityParams
from relight_engine.engine import relight
from relight_engine.geometry import (
    backproject_depth,
    clean_depth_background_skirt,
    compute_depth_normal_consistency,
    extract_micro_normal_bump,
    integrate_normals_with_depth_prior,
)
from relight_engine.lighting import evaluate_rig
from relight_engine.shadows import ray_visibility


def test_poisson_normal_depth_integration_aligns_normals():
    h, w = 64, 64
    cam = CameraModel(fx=120.0, fy=120.0, cx=32.0, cy=32.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (18.0 ** 2)
    bump = np.exp(-r2) * 0.25
    z_true = 2.0 - bump

    pts_true = backproject_depth(z_true, cam)
    # Compute smooth analytical-like normals from true surface
    dz_dx = np.gradient(z_true, axis=1)
    dz_dy = np.gradient(z_true, axis=0)
    nx = (cam.fx / z_true) * dz_dx
    ny = -(cam.fy / z_true) * dz_dy
    nz = -np.ones_like(z_true)
    normals = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)

    # Start with overly flat/coarse depth prior
    z_coarse = np.full((h, w), 2.0, dtype=np.float32) - 0.15 * bump
    z_fused = integrate_normals_with_depth_prior(z_coarse, normals, cam, screen_lambda=0.005)
    pts_fused = backproject_depth(z_fused, cam)
    cons = compute_depth_normal_consistency(pts_fused, z_fused, normals)
    assert cons["mean_angle_deg"] < 8.0


def test_clean_depth_background_skirt_restores_leaked_bg():
    h, w = 80, 80
    depth = np.full((h, w), 4.5, dtype=np.float32)
    rgb = np.full((h, w, 3), 0.04, dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    fg = ((xx - 40) ** 2 + (yy - 40) ** 2) <= 14 ** 2
    skirt = (((xx - 40) ** 2 + (yy - 40) ** 2) <= 22 ** 2) & (~fg)

    depth[fg] = 1.8
    depth[skirt] = 2.1  # Monocular depth leaked near onto dark background
    rgb[fg] = 0.75

    cleaned = clean_depth_background_skirt(depth, rgb, semantic_masks={"object": fg.astype(np.float32)})
    assert float(np.median(cleaned[skirt])) >= 3.8
    assert float(np.median(cleaned[fg])) <= 1.9

    # Medium-bright/textured backdrop + shaded skin without a hand mask must remain untouched
    med_bg_depth = depth.copy()
    med_bg_rgb = np.full((h, w, 3), 0.35, dtype=np.float32)
    med_bg_rgb[fg] = 0.65
    cleaned_med = clean_depth_background_skirt(med_bg_depth, med_bg_rgb, semantic_masks=None)
    assert np.allclose(cleaned_med, med_bg_depth, atol=1e-6)


def test_micro_normal_bump_preserves_unit_length_and_responds_to_texture():
    h, w = 48, 48
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tex = 0.5 + 0.15 * np.sin(xx * 0.8) * np.cos(yy * 0.8)
    rgb = np.repeat(tex[..., None], 3, axis=-1).astype(np.float32)

    bumped = extract_micro_normal_bump(normals, rgb, strength=0.30)
    norms = np.linalg.norm(bumped, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert float(np.std(bumped[..., 0])) > 0.001

    flat_rgb = np.full((h, w, 3), 0.5, dtype=np.float32)
    flat_bumped = extract_micro_normal_bump(normals, flat_rgb, strength=0.30)
    assert np.allclose(flat_bumped, normals, atol=1e-6)


def test_stratified_area_shadow_samples_produce_smooth_penumbra():
    h, w = 64, 64
    cam = CameraModel(fx=100.0, fy=100.0, cx=32.0, cy=32.0)
    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[20:36, 20:30] = 1.8  # Foreground blocker
    pts = backproject_depth(depth, cam)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0

    light_pos = np.array([-1.5, 0.2, 0.5], dtype=np.float32)
    to_light = light_pos[None, None, :] - pts
    max_dist = np.linalg.norm(to_light, axis=-1)
    settings = EngineSettings.studio_3d(area_shadow_samples=4, shadow_steps=20)
    vis = ray_visibility(
        pts, normals, depth, cam, to_light, max_dist, "relative", settings, light_radius=0.30
    )
    assert vis.shape == (h, w)
    assert np.isfinite(vis).all()
    assert 0.0 <= float(vis.min()) <= float(vis.max()) <= 1.0
    # Cast shadow region on background right of blocker should have partial/full occlusion
    assert float(np.min(vis[22:34, 31:48])) < 0.85


def test_virtual_studio_reflection_probe_follows_reflection_vector():
    h, w = 64, 64
    cam = CameraModel(fx=120.0, fy=120.0, cx=32.0, cy=32.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / (20.0 ** 2)
    depth = 2.5 - 0.35 * np.exp(-r2)
    pts = backproject_depth(depth, cam)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 0] = (xx - 32.0) / 28.0
    normals[..., 1] = -(yy - 32.0) / 28.0
    normals[..., 2] = -np.sqrt(np.maximum(1.0 - normals[..., 0] ** 2 - normals[..., 1] ** 2, 0.05))
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)

    rgb = np.full((h, w, 3), 0.35, dtype=np.float32)
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=depth,
        depth_kind="relative",
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        normals=normals,
        albedo=np.full((h, w, 3), 0.75, dtype=np.float32),
        roughness=np.full((h, w), 0.14, dtype=np.float32),
        metalness=np.full((h, w), 0.92, dtype=np.float32),
        specular_level=np.full((h, w), 0.90, dtype=np.float32),
    )
    rig = LightingRig(
        lights=[Light(name="red_pt", kind="point", position=(-1.8, 0.0, 1.0), color=(1.0, 0.15, 0.15), strength=3.0, radius=0.35)],
        ambient_strength=0.10,
    )
    settings_on = EngineSettings.studio_3d(virtual_studio_reflection=True)
    settings_off = EngineSettings.studio_3d(virtual_studio_reflection=False)
    _, spec_on, _ = evaluate_rig(scene, cam, pts, normals, rig, settings_on)
    _, spec_off, _ = evaluate_rig(scene, cam, pts, normals, rig, settings_off)
    # Virtual studio reflection probe should add significant red reflection energy on the left-facing curved flank
    assert float(np.mean(spec_on[:, :32, 0])) > float(np.mean(spec_off[:, :32, 0])) + 0.03
    assert float(np.mean(spec_on[:, :32, 0])) > float(np.mean(spec_on[:, 32:, 0]))


def test_auto_conductor_calibration_detects_high_contrast_neutral_metal():
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Smooth Gaussian chrome specular rolloff from dark mirror trough (0.05) to bright highlight (0.93)
    chrome_bar = 0.05 + 0.88 * np.exp(-((xx - 32.0) / 5.5) ** 2)
    rgb = np.full((h, w, 3), 0.05, dtype=np.float32)
    rgb[16:46, :, :] = chrome_bar[16:46, :, None]
    # Skin-masked region with same specular rolloff should NOT be marked metallic
    skin_mask = np.zeros((h, w), dtype=np.float32)
    skin_mask[48:, :] = 1.0
    rgb[48:60, :, :] = chrome_bar[48:60, :, None]

    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_kind="relative",
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        semantic_masks={"skin": skin_mask},
    )
    res = calibrate_conductor_from_appearance(scene)
    assert res is not None
    metalness, roughness = res
    assert float(np.max(metalness[20:44, 24:40])) >= 0.70
    assert float(np.min(roughness[20:44, 24:40])) <= 0.28
    assert float(np.max(metalness[52:60, 28:36])) < 0.25


def test_skin_likelihood_detects_face_and_hand_while_ignoring_metal():
    h, w = 80, 96
    rgb = np.full((h, w, 3), 0.12, dtype=np.float32)
    depth = np.full((h, w), 4.0, dtype=np.float32)

    # Smooth skin patch inside 'man' mask
    rgb[10:28, 12:30] = np.array([0.62, 0.44, 0.35], dtype=np.float32)
    # Smooth palm skin around a metallic watch in foreground
    rgb[40:72, 12:40] = np.array([0.55, 0.39, 0.33], dtype=np.float32)
    depth[38:74, 10:42] = 1.6

    # Metallic watch bezel on the hand (neutral shiny + watch mask)
    rgb[50:62, 20:32] = np.array([0.82, 0.82, 0.84], dtype=np.float32)
    metalness = np.zeros((h, w), dtype=np.float32)
    metalness[50:62, 20:32] = 0.90

    # False-positive patches in separate regions: textured wood, yellow sand, warm cardboard, pebbled leather, shiny copper
    rng = np.random.default_rng(7)
    rgb[10:26, 48:64] = np.clip(0.45 + 0.08 * rng.standard_normal((16, 16, 1)), 0.0, 1.0) * np.array([0.65, 0.42, 0.28], dtype=np.float32)  # textured wood
    rgb[10:26, 70:86] = np.clip(np.array([0.78, 0.68, 0.42]) + 0.07 * rng.standard_normal((16, 16, 3)), 0.0, 1.0)  # sand
    rgb[34:50, 48:64] = np.array([0.64, 0.50, 0.36], dtype=np.float32)  # cardboard (no person/fg mask)
    rgb[34:50, 70:86] = np.clip(np.array([0.48, 0.30, 0.22]) + 0.09 * rng.standard_normal((16, 16, 1)), 0.0, 1.0)  # leather
    rgb[56:72, 56:76] = np.array([0.78, 0.46, 0.32], dtype=np.float32)  # copper
    metalness[56:72, 56:76] = 0.88

    man_mask = np.zeros((h, w), dtype=np.float32)
    man_mask[8:32, 10:34] = 1.0
    watch_mask = np.zeros((h, w), dtype=np.float32)
    watch_mask[50:62, 20:32] = 1.0
    cardboard_mask = np.zeros((h, w), dtype=np.float32)
    cardboard_mask[34:50, 48:64] = 1.0

    skin_w = compute_skin_likelihood_map(
        rgb,
        semantic_masks={"man": man_mask, "watch": watch_mask, "cardboard": cardboard_mask},
        depth=depth,
        material_maps={"metalness": metalness},
        min_skin_confidence=0.35,
    )
    assert float(np.mean(skin_w[14:24, 16:26])) > 0.65  # face inside man
    assert float(np.mean(skin_w[43:48, 16:36])) > 0.50  # hand around watch
    assert float(np.max(skin_w[52:60, 22:30])) < 0.05   # metallic watch zeroed
    assert float(np.max(skin_w[10:26, 48:64])) < 0.05   # wood
    assert float(np.max(skin_w[10:26, 70:86])) < 0.05   # sand
    assert float(np.max(skin_w[34:50, 48:64])) < 0.05   # cardboard
    assert float(np.max(skin_w[34:50, 70:86])) < 0.05   # leather
    assert float(np.max(skin_w[56:72, 56:76])) < 0.05   # copper


def _make_skin_and_metal_scene():
    h, w = 64, 96
    cam = CameraModel(90.0, 90.0, w / 2.0, h / 2.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    rgb = np.full((h, w, 3), 0.25, dtype=np.float32)
    tex = 0.03 * np.sin(xx * 1.3) * np.cos(yy * 1.3)
    # Left curved skin sphere inside 'face' mask
    r2_s = ((xx - 28.0) ** 2 + (yy - 32.0) ** 2) / (16.0 ** 2)
    skin_reg = r2_s <= 1.0
    rgb[skin_reg] = np.clip(
        np.array([0.56, 0.40, 0.33], dtype=np.float32)[None, :] + tex[skin_reg, None], 0.05, 0.95
    )
    # Right metallic cylinder outside skin mask
    metal_reg = (xx >= 58) & (xx <= 80) & (yy >= 14) & (yy <= 50)
    rgb[metal_reg] = np.array([0.45, 0.46, 0.48], dtype=np.float32)

    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[skin_reg] = 2.0 - 0.3 * (1.0 - r2_s[skin_reg])
    depth[metal_reg] = 1.9 - 0.25 * (1.0 - ((xx[metal_reg] - 69.0) / 11.0) ** 2)

    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    normals[skin_reg, 0] = (xx[skin_reg] - 28.0) / 18.0
    normals[skin_reg, 1] = -(yy[skin_reg] - 32.0) / 18.0
    normals[skin_reg, 2] = -np.sqrt(np.maximum(1.0 - normals[skin_reg, 0] ** 2 - normals[skin_reg, 1] ** 2, 0.08))
    normals[metal_reg, 0] = (xx[metal_reg] - 69.0) / 13.0
    normals[metal_reg, 2] = -np.sqrt(np.maximum(1.0 - normals[metal_reg, 0] ** 2, 0.08))

    face_mask = np.zeros((h, w), dtype=np.float32)
    face_mask[14:50, 10:46] = 1.0
    metal_mask = metal_reg.astype(np.float32)
    metalness = np.zeros((h, w), dtype=np.float32)
    metalness[metal_reg] = 0.90
    roughness = np.full((h, w), 0.55, dtype=np.float32)
    roughness[metal_reg] = 0.14
    albedo = rgb.copy()
    src_shading = np.full((h, w), 0.70, dtype=np.float32)
    src_shading[skin_reg] = 0.42  # simulate IID skin shading underestimate

    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=depth,
        depth_kind="relative",
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        albedo=albedo,
        source_shading=src_shading,
        metalness=metalness,
        roughness=roughness,
        semantic_masks={"face": face_mask, "chrome": metal_mask},
    )
    src_rig = LightingRig(ambient_strength=0.5, lights=(Light("src", "directional", direction=(0, 0, -1), strength=0.5, casts_shadows=False),))
    tgt_rig = LightingRig(
        ambient_color=(0.25, 0.25, 0.28),
        ambient_strength=0.18,
        lights=(
            Light("k1", "point", position=(-1.4, 0.2, 0.8), color=(1.0, 0.78, 0.55), strength=3.0, radius=0.25),
            Light("k2", "point", position=(1.4, 0.1, 0.9), color=(0.45, 0.70, 1.0), strength=2.6, radius=0.25),
        ),
    )
    return scene, cam, src_rig, tgt_rig, skin_reg, metal_reg


def test_skin_fidelity_guard_preserves_skin_exposure_and_texture_energy():
    scene, cam, src_rig, tgt_rig, skin_reg, metal_reg = _make_skin_and_metal_scene()
    s_off = EngineSettings.studio_3d(skin_fidelity=SkinFidelityParams(enabled=False))
    s_on = EngineSettings.studio_3d(skin_fidelity=SkinFidelityParams(enabled=True))
    res_off = relight(scene, cam, src_rig, tgt_rig, s_off)
    res_on = relight(scene, cam, src_rig, tgt_rig, s_on)

    # Guard reduces over-brightening on skin relative to guard-off and preserves texture band
    skin_mean_orig = float(np.mean(scene.rgb_srgb[skin_reg]))
    skin_mean_off = float(np.mean(res_off.rgb_srgb[skin_reg]))
    skin_mean_on = float(np.mean(res_on.rgb_srgb[skin_reg]))
    assert skin_mean_on <= skin_mean_off
    assert abs(skin_mean_on - skin_mean_orig) < 0.10

    def _lap(img):
        g = img.mean(axis=-1)
        l = g[1:-1, 1:-1] * 4.0 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        m = skin_reg[1:-1, 1:-1]
        return float(np.sqrt(np.mean(l[m] ** 2)))

    lap_ratio = _lap(res_on.rgb_srgb) / max(_lap(scene.rgb_srgb), 1e-6)
    assert 0.85 <= lap_ratio <= 1.15
    # Metal keeps full specular contrast unchanged
    assert float(np.std(res_on.rgb_srgb[metal_reg])) >= float(np.std(res_off.rgb_srgb[metal_reg])) - 1e-5


def test_non_skin_pixels_identical_with_guard():
    scene, cam, src_rig, tgt_rig, _, _ = _make_skin_and_metal_scene()
    sw = compute_skin_likelihood_map(scene.rgb_srgb, scene.semantic_masks, scene.depth, {"metalness": scene.metalness})
    s_off = EngineSettings.studio_3d(skin_fidelity=SkinFidelityParams(enabled=False))
    s_on = EngineSettings.studio_3d(skin_fidelity=SkinFidelityParams(enabled=True))
    res_off = relight(scene, cam, src_rig, tgt_rig, s_off)
    res_on = relight(scene, cam, src_rig, tgt_rig, s_on)
    non_skin = sw == 0.0
    assert float(np.max(np.abs(res_on.rgb_srgb[non_skin] - res_off.rgb_srgb[non_skin]))) < 1e-4


def test_noop_relight_is_identity():
    scene, cam, src_rig, _, _, _ = _make_skin_and_metal_scene()
    s_on = EngineSettings.studio_3d(skin_fidelity=SkinFidelityParams(enabled=True))
    res = relight(scene, cam, src_rig, src_rig, s_on)
    assert float(np.max(np.abs(res.rgb_srgb - scene.rgb_srgb))) < 1e-4


def test_mask_edge_continuity():
    scene, cam, src_rig, tgt_rig, _, _ = _make_skin_and_metal_scene()
    sw = compute_skin_likelihood_map(scene.rgb_srgb, scene.semantic_masks, scene.depth, {"metalness": scene.metalness})
    s_off = EngineSettings.studio_3d(skin_fidelity=SkinFidelityParams(enabled=False))
    s_on = EngineSettings.studio_3d(skin_fidelity=SkinFidelityParams(enabled=True))
    res_off = relight(scene, cam, src_rig, tgt_rig, s_off)
    res_on = relight(scene, cam, src_rig, tgt_rig, s_on)
    # Guard delta across feathered mask boundary (0 < sw < 1) must transition smoothly without a seam/halo
    edge_band = (sw > 0.05) & (sw < 0.95)
    assert np.any(edge_band)
    delta = res_on.rgb_srgb - res_off.rgb_srgb
    dx_delta = np.abs(np.diff(delta, axis=1, prepend=delta[:, :1]))
    assert float(np.percentile(dx_delta[edge_band], 95)) < 0.04
