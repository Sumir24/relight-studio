"""Milestone M3 (Phase 6) tests: detail/illumination separation, highlight bounds, display transform, 16-bit I/O, and preservation report."""
from __future__ import annotations

import numpy as np
import pytest

from relight_engine import (
    ALPHA_CONVENTION,
    CameraModel,
    EngineSettings,
    Light,
    LightingRig,
    SceneBuffers,
    evaluate_preservation,
    load_image,
    relight,
    save_image,
)
from relight_engine.composite import _compute_detail_lock_weight_map, compose_hybrid_neural


def test_phase6_detail_weight_separates_illumination_edges_from_albedo_texture():
    h, w = 32, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.full((h, w, 3), 0.5, dtype=np.float32)
    albedo = np.full((h, w, 3), 0.5, dtype=np.float32)
    albedo[:, 22:] = 0.25  # sharp surface albedo edge on right

    source_shading = np.full((h, w), 0.85, dtype=np.float32)
    source_shading[:, :10] = 0.25  # sharp source illumination shadow edge on left

    res_lobe = np.zeros((h, w, 3), dtype=np.float32)
    res_lobe[14:18, 14:18] = 0.25  # confident baked glare artifact in center

    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        depth_kind="relative",
        albedo=albedo,
        source_shading=source_shading,
    )
    geom_conf = np.ones((h, w), np.float32)
    ratio = np.full((h, w), 1.8, np.float32)
    layers = (np.zeros_like(rgb), np.zeros_like(rgb), res_lobe)
    w_map = _compute_detail_lock_weight_map(scene, geom_conf, ratio, layers, EngineSettings())

    # Confident illumination boundary & baked glare drop wavelet detail copying; true albedo edge stays locked.
    assert float(w_map[16, 10]) < 0.70
    assert float(w_map[16, 16]) < 0.50
    assert float(w_map[16, 22]) > 0.90


def test_phase6_bounded_highlight_relocation_and_protected_display_transform():
    h, w = 24, 24
    rgb = np.full((h, w, 3), 0.45, dtype=np.float32)
    prot = np.zeros((h, w), dtype=np.float32)
    prot[:, :6] = 1.0
    alpha = np.ones((h, w), dtype=np.float32)
    alpha[:4, :] = 0.0
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0

    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=np.full((h, w), 2.0, np.float32),
        depth_confidence=np.ones((h, w), np.float32),
        lightable_mask=np.ones((h, w), np.float32),
        protected_mask=prot,
        alpha_mask=alpha,
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), np.float32),
        albedo=rgb.copy(),
        source_shading=np.full((h, w), 0.5, np.float32),
        roughness=np.full((h, w), 0.12, np.float32),
    )
    cam = CameraModel(60.0, 60.0, w / 2.0, h / 2.0)
    src = LightingRig(ambient_strength=0.4)
    # Extreme highlight relocation + filmic shoulder should remain bounded and bit-exact on protected/transparent pixels.
    tgt = LightingRig(
        ambient_strength=0.4,
        lights=(Light("glare", "point", position=(0.0, 0.0, 0.2), strength=50.0, casts_shadows=False),),
    )
    res = relight(
        scene, cam, src, tgt,
        EngineSettings(
            renderer_mode="intrinsic",
            highlight_relocation_strength=1.0,
            filmic_highlight_rolloff=True,
            specular_scale=3.0,
        ),
    )
    assert np.all(np.isfinite(res.rgb_srgb))
    assert float(np.max(res.rgb_srgb)) <= 1.0
    pres = res.metrics["preservation"]
    assert pres["alpha_convention"] == ALPHA_CONVENTION
    assert pres["protected_max_abs_error"] == 0.0
    assert pres["transparent_max_abs_error"] == 0.0
    assert pres["dimensions_preserved"] is True

    # Even compose_hybrid_neural preserves protected pixels bit-exactly through highlight rolloff.
    neu_out = compose_hybrid_neural(scene, np.full_like(rgb, 0.95), np.ones((h, w), np.float32), EngineSettings(filmic_highlight_rolloff=True))
    neu_pres = evaluate_preservation(scene, neu_out, target_alpha=alpha)
    assert neu_pres["protected_max_abs_error"] == 0.0
    assert neu_pres["transparent_max_abs_error"] == 0.0


def test_phase6_16bit_png_roundtrip_and_straight_alpha(tmp_path):
    h, w = 16, 32
    # Fine 16-bit ramp below 1/255 step precision
    ramp = np.linspace(0.200, 0.212, w, dtype=np.float32)[None, :, None].repeat(h, axis=0).repeat(3, axis=2)
    alpha = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    p16 = tmp_path / "grad16.png"
    save_image(p16, ramp, alpha_mask=alpha, bit_depth=16)

    loaded = load_image(p16)
    assert loaded.metadata["bit_depth"] == 16
    assert loaded.metadata["alpha_convention"] == "straight"
    assert loaded.alpha_mask is not None
    # 16-bit precision preserves fine tonal ramps cleanly without 8-bit 1/255 banding
    assert np.allclose(loaded.rgb_srgb, ramp, atol=5e-5)
    assert np.allclose(loaded.alpha_mask, alpha, atol=5e-5)
