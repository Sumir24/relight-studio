"""M2 Release Gate tests: Phase 3 SourceIllumination, Phase 4 multi-light/BRDF, Phase 5 geometry/shadows/SSGI."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from relight_engine import (
    CameraModel,
    EngineSettings,
    Light,
    LightingRig,
    SceneBuffers,
    SourceIllumination,
    SourceLightCalibration,
    compute_intrinsic_reconstruction_error,
    estimate_source_illumination,
    relight,
)
from relight_engine.color import linear_to_srgb
from relight_engine.geometry import (
    backproject_depth,
    compute_depth_normal_consistency,
    prepare_geometry,
    refine_depth_to_rgb_edges,
)
from relight_engine.lighting import (
    _area_samples,
    _prefilter_equirectangular_env,
    compute_indirect_bounce,
    evaluate_rig,
)
from relight_engine.workflows import source_illumination_from_dict


def test_phase3_source_illumination_contract_fitting_outliers_and_reconstruction_error():
    h, w = 36, 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)

    # Hemisphere normals on center dome so SH can resolve 3D key direction cleanly
    dx = (xx - w / 2.0) / 14.0
    dy = (h / 2.0 - yy) / 14.0
    r2 = dx * dx + dy * dy
    nz = -np.sqrt(np.maximum(1.0 - np.minimum(r2, 0.95), 0.05))
    normals = np.stack([dx * (r2 < 0.95), dy * (r2 < 0.95), nz], axis=-1).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)

    true_dir = np.array([-0.6, 0.35, -0.717], dtype=np.float32)
    true_dir /= np.linalg.norm(true_dir)
    true_shading = 0.25 + 0.85 * np.clip(np.sum(normals * true_dir[None, None, :], axis=-1), 0.0, 1.0)

    albedo_lin = np.full((h, w, 3), 0.45, dtype=np.float32)
    src_lin = np.clip(albedo_lin * true_shading[..., None], 0.01, 0.99)
    rgb_srgb = linear_to_srgb(src_lin)

    # Inject clipped highlights, transparent border, uncertain patch, and emissive spikes
    rgb_srgb[:4, :] = 0.99
    shading_corrupt = true_shading.copy()
    shading_corrupt[:4, :] = 12.0  # emissive/clipped outlier
    alpha = np.ones((h, w), dtype=np.float32)
    alpha[:, :4] = 0.0
    mat_conf = np.ones((h, w), dtype=np.float32)
    mat_conf[-6:, :] = 0.05

    scene = SceneBuffers(
        rgb_srgb=rgb_srgb,
        depth=np.full((h, w), 2.5, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        alpha_mask=alpha,
        material_confidence=mat_conf,
        albedo=linear_to_srgb(albedo_lin),
        source_shading=shading_corrupt,
    )

    proposal = estimate_source_illumination(scene, normals)
    assert proposal.provenance["reviewed"] is False
    assert proposal.sh_coefficients is not None and proposal.sh_coefficients.shape == (9, 3)
    est_dir = np.asarray(proposal.lights[0].direction, dtype=np.float32)
    assert float(np.dot(est_dir, true_dir)) > 0.78

    err = compute_intrinsic_reconstruction_error(scene)
    assert err is not None
    assert err["mean_error"] > 0.0

    # Serialization / dict parsing + add vs replace operations
    single = SourceLightCalibration(key_direction=(0, 0, -1), key_strength=0.6)
    si = SourceIllumination.from_single_key(single)
    parsed = source_illumination_from_dict({
        "ambient_strength": 0.2,
        "exposure_scale": 1.1,
        "white_balance": [1.0, 0.95, 0.9],
        "lights": [{"kind": "directional", "direction": [0.0, 0.0, -1.0], "strength": 0.7}],
    })
    assert len(parsed.to_rig().lights) == 1
    assert si.to_rig().lights[0].strength == pytest.approx(0.6)

    tgt = LightingRig(ambient_strength=0.1, lights=(Light("k", "directional", direction=(-0.5, 0.1, -0.86), strength=1.0, casts_shadows=False),))
    res_rep = relight(scene, cam, si.to_rig(), tgt, EngineSettings(operation="replace"))
    res_add = relight(scene, cam, si.to_rig(), tgt, EngineSettings(operation="add"))
    assert float(np.mean(res_add.rgb_srgb)) > float(np.mean(res_rep.rgb_srgb)) + 0.04


def test_phase4_area_sampling_spotlights_clearcoat_attenuation_and_weighted_shadows():
    h, w = 32, 48
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[:, 20:24] = 1.2  # casts shadow on right half from left key
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)

    # 1. Centered single-sample area light & explicit orientation
    area1 = Light("a1", "area", position=(0.0, 0.0, 1.0), direction=(0.0, 0.0, 1.0), area_size=(1.0, 1.0), samples=1)
    pts = _area_samples(area1)
    assert len(pts) == 1 and np.allclose(pts[0], [0.0, 0.0, 1.0], atol=1e-6)

    area4 = Light("a4", "area", position=(0.0, 0.0, 1.0), direction=(0.0, 0.0, 1.0), area_size=(1.0, 1.0), samples=4)
    pts4 = _area_samples(area4)
    assert len(pts4) == 4 and np.allclose(np.mean(pts4, axis=0), [0.0, 0.0, 1.0], atol=1e-6)

    # 2. Spotlight cone falloff: center bright, wide corner unlit
    scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.45, dtype=np.float32),
        depth=depth,
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
    )
    spot = Light(
        "sp", "spot", position=(0.0, 0.0, 1.0), direction=(0.0, 0.0, 1.0),
        spot_inner_angle_deg=12.0, spot_outer_angle_deg=22.0, strength=3.0, casts_shadows=False,
    )
    pts3d = backproject_depth(depth, cam)
    d_spot, _, _ = evaluate_rig(scene, cam, pts3d, normals, LightingRig(ambient_strength=0.05, lights=(spot,)), EngineSettings())
    assert float(np.mean(d_spot[h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2])) > float(np.mean(d_spot[0:3, 0:3])) * 3.0

    # 3. Physical vs artistic intensity mode
    pt_art = Light("p_art", "point", position=(0.0, 0.0, 1.0), strength=2.0, intensity_mode="artistic", casts_shadows=False)
    pt_phy = Light("p_phy", "point", position=(0.0, 0.0, 1.0), strength=2.0, intensity_mode="physical", casts_shadows=False)
    d_art, _, _ = evaluate_rig(scene, cam, pts3d, normals, LightingRig(ambient_strength=0.0, lights=(pt_art,)), EngineSettings())
    d_phy, _, _ = evaluate_rig(scene, cam, pts3d, normals, LightingRig(ambient_strength=0.0, lights=(pt_phy,)), EngineSettings())
    assert np.allclose(d_art / np.pi, d_phy, atol=1e-4)

    # 4. Contribution-weighted shadow diagnostics & per-light records with fill light
    key_l = Light("key", "directional", direction=(-0.85, 0.0, -0.52), strength=1.5, casts_shadows=True)
    fill_r = Light("fill", "directional", direction=(0.8, 0.0, -0.6), strength=1.0, casts_shadows=False)
    res = relight(scene, cam, LightingRig(ambient_strength=0.5), LightingRig(ambient_strength=0.1, lights=(key_l, fill_r)))
    assert len(res.metrics["per_light_contributions"]) == 2
    assert res.metrics["shadow_visibility_weighted_mean"] > res.metrics["shadow_visibility_mean"]

    # 5. Clearcoat attenuates underlying base specular/grazing response rather than strictly adding
    grazing_normals = np.zeros((h, w, 3), dtype=np.float32)
    grazing_normals[..., 0] = 0.70
    grazing_normals[..., 2] = -0.71
    sc_base = SceneBuffers(
        rgb_srgb=scene.rgb_srgb, depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=scene.depth_confidence, lightable_mask=scene.lightable_mask,
        depth_kind="relative", normals=grazing_normals, geometry_confidence=scene.geometry_confidence,
        roughness=np.full((h, w), 0.35, np.float32), clearcoat=np.zeros((h, w), np.float32),
    )
    sc_coat = SceneBuffers(
        rgb_srgb=scene.rgb_srgb, depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=scene.depth_confidence, lightable_mask=scene.lightable_mask,
        depth_kind="relative", normals=grazing_normals, geometry_confidence=scene.geometry_confidence,
        roughness=np.full((h, w), 0.35, np.float32), clearcoat=np.ones((h, w), np.float32),
    )
    d0, _, _ = evaluate_rig(sc_base, cam, backproject_depth(sc_base.depth, cam), grazing_normals, LightingRig(ambient_strength=0.1, lights=(fill_r,)), EngineSettings())
    d1, _, _ = evaluate_rig(sc_coat, cam, backproject_depth(sc_coat.depth, cam), grazing_normals, LightingRig(ambient_strength=0.1, lights=(fill_r,)), EngineSettings())
    assert float(np.mean(d1)) < float(np.mean(d0))


def test_phase5_depth_normal_consistency_texture_gated_refinement_and_ssgi_desaturation():
    h, w = 24, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = np.full((h, w), 2.0, np.float32)
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    pts = backproject_depth(depth, cam)

    bad_normals = np.zeros((h, w, 3), dtype=np.float32)
    bad_normals[..., 0] = 0.8
    bad_normals[..., 2] = -0.6
    cons = compute_depth_normal_consistency(pts, depth, bad_normals)
    assert cons["mean_angle_deg"] > 40.0

    # Unauthored geometry_confidence incorporates depth-normal disagreement
    sc_bad = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.5, np.float32),
        depth=depth,
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
        normals=bad_normals,
        geometry_confidence=None,
    )
    _, _, gconf = prepare_geometry(sc_bad, cam)
    assert float(np.mean(gconf)) < 0.85

    # Flat depth with sharp 2D RGB stripes keeps flat depth intact
    stripes_rgb = np.clip(0.5 + 0.4 * np.sin(xx * 1.2)[..., None], 0.0, 1.0).repeat(3, axis=-1).astype(np.float32)
    refined = refine_depth_to_rgb_edges(depth, stripes_rgb)
    assert np.allclose(refined, depth, atol=1e-4)

    # Indirect bounce without albedo desaturates vivid source color cast & stays bounded
    cast_rgb = np.full((h, w, 3), np.array([0.85, 0.15, 0.15], np.float32), dtype=np.float32)
    sc_cast = SceneBuffers(
        rgb_srgb=cast_rgb, depth=depth,
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
    )
    normals_z = np.zeros((h, w, 3), dtype=np.float32)
    normals_z[..., 2] = -1.0
    bounce = compute_indirect_bounce(sc_cast, pts, normals_z, np.ones((h, w, 3), np.float32), EngineSettings(indirect_bounce_strength=1.0))
    # Bounce chromatic ratio must be less red-saturated than raw cast RGB
    raw_rb = float(cast_rgb[0, 0, 0] / cast_rgb[0, 0, 2])
    bounce_rb = float(np.mean(bounce[..., 0]) / max(float(np.mean(bounce[..., 2])), 1e-6))
    assert bounce_rb < raw_rb

    # Cached equirectangular IBL prefiltering returns identical arrays across repeated calls
    env = np.full((16, 32, 3), 0.35, dtype=np.float32)
    m1, i1 = _prefilter_equirectangular_env(env)
    m2, i2 = _prefilter_equirectangular_env(env)
    assert m1 is m2 and i1 is i2
