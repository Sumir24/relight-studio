"""Tests for Tier 1 real-image-preserving relighting and à-trous wavelet identity locking."""
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
    prepare_relight_session,
    relight,
)
from relight_engine.adapters.protocols import NeuralRelightPrediction
from relight_engine.color import linear_to_srgb, luminance, srgb_to_linear
from relight_engine.wavelet import atrous_decompose, atrous_reconstruct, compose_wavelet_identity_lock


def test_atrous_wavelet_exact_reconstruction():
    rng = np.random.default_rng(11)
    img_2d = rng.uniform(0.05, 0.95, size=(40, 56)).astype(np.float32)
    bands, low = atrous_decompose(img_2d, levels=4)
    assert len(bands) == 4
    recon = atrous_reconstruct(bands, low)
    assert np.allclose(recon, img_2d, atol=1e-5)

    img_3d = rng.uniform(0.05, 0.95, size=(32, 48, 3)).astype(np.float32)
    bands3, low3 = atrous_decompose(img_3d, levels=3)
    recon3 = atrous_reconstruct(bands3, low3)
    assert np.allclose(recon3, img_3d, atol=1e-5)


def test_wavelet_identity_lock_keeps_hf_texture_and_chroma_under_smooth_relight():
    h, w = 48, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src = np.full((h, w, 3), 0.36, dtype=np.float32)
    # Fine 2-3px luminance weave + fine horizontal chromatic pinstripe
    src += (0.07 * np.sin(xx * 1.9) * np.cos(yy * 1.7))[..., None]
    src[..., 0] += 0.06 * np.sin(xx * 1.5)
    src[..., 2] -= 0.06 * np.sin(xx * 1.5)
    src = np.clip(src, 0.02, 0.95)

    # Smooth warm-lit low-freq neural output without fine texture
    grad = np.linspace(0.70, 0.22, w, dtype=np.float32)[None, :, None]
    relit = np.broadcast_to(grad * np.array([[[1.10, 0.88, 0.72]]], dtype=np.float32), (h, w, 3)).copy()

    locked = compose_wavelet_identity_lock(src, relit, levels=4, detail_lock_strength=0.95, mid_blend=0.55)
    assert np.isfinite(locked).all()
    # Left stays bright and warm, right stays darker (macro lighting preserved)
    assert float(luminance(locked[:, 8:16]).mean()) > float(luminance(locked[:, 48:56]).mean()) + 0.15
    # Fine HF spatial variance survives in both lit and darker areas
    assert float(np.std(locked[:, 10:26, 0] - np.mean(locked[:, 10:26, 0], axis=0, keepdims=True))) > 0.025


def test_engine_settings_tier1_validation_and_real_preserving_preset():
    preset = EngineSettings.real_preserving()
    assert preset.detail_preservation_mode == "wavelet"
    assert 0.0 < preset.color_transfer_strength < 1.0
    assert 0.0 < preset.background_light_influence < 1.0
    assert 0.0 < preset.real_image_preservation < 1.0
    assert preset.auto_sh_source is True

    with pytest.raises(ValueError, match="color_transfer_strength"):
        EngineSettings(color_transfer_strength=1.5).validate()
    with pytest.raises(ValueError, match="background_light_influence"):
        EngineSettings(background_light_influence=-0.1).validate()
    with pytest.raises(ValueError, match="real_image_preservation"):
        EngineSettings(real_image_preservation=1.2).validate()
    with pytest.raises(ValueError, match="highlight_relocation_strength"):
        EngineSettings(highlight_relocation_strength=-0.2).validate()
    with pytest.raises(ValueError, match="wavelet_levels"):
        EngineSettings(wavelet_levels=0).validate()


def test_tier1_constrained_ratio_and_background_attenuation_preserve_real_scene():
    h, w = 48, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Left foreground subject (z=1.5), right background wall (z=4.2)
    depth = np.where(xx < w // 2, 1.5, 4.2).astype(np.float32)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    bg_mask = np.where(xx >= w // 2, 1.0, 0.0).astype(np.float32)
    fg_mask = 1.0 - bg_mask

    # Real photo: neutral textured wall + detailed gray subject
    tex = 0.04 * np.sin(xx * 1.4) * np.cos(yy * 1.4)
    rgb = np.clip(0.52 + tex[..., None], 0.05, 0.95).repeat(3, axis=-1).astype(np.float32)
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=depth,
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        semantic_masks={"subject": fg_mask, "background": bg_mask},
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    src_rig = LightingRig(ambient_strength=0.65, lights=(
        Light("src", "directional", direction=(0.0, 0.0, -1.0), strength=0.6, casts_shadows=False),
    ))
    red_rig = LightingRig(ambient_color=(0.15, 0.05, 0.05), ambient_strength=0.10, lights=(
        Light("red_left", "point", position=(-1.2, 0.0, 1.0), color=(1.0, 0.04, 0.06), strength=4.0, casts_shadows=False),
    ))

    res_old = relight(scene, camera, src_rig, red_rig, EngineSettings(renderer_mode="ratio"))
    res_new = relight(scene, camera, src_rig, red_rig, EngineSettings.real_preserving(renderer_mode="ratio"))

    # Background stays much closer to the real neutral wall instead of turning neon red
    bg_red_shift_old = float(np.mean(res_old.rgb_srgb[:, w // 2 + 6 :, 0] - res_old.rgb_srgb[:, w // 2 + 6 :, 2]))
    bg_red_shift_new = float(np.mean(res_new.rgb_srgb[:, w // 2 + 6 :, 0] - res_new.rgb_srgb[:, w // 2 + 6 :, 2]))
    assert bg_red_shift_new < bg_red_shift_old * 0.55

    # Foreground still receives clear red relighting on the left, while preserving texture
    fg_red_shift = float(np.mean(res_new.rgb_srgb[:, 4 : w // 2 - 4, 0] - res_new.rgb_srgb[:, 4 : w // 2 - 4, 2]))
    assert fg_red_shift > 0.08

    # Exact noop holds even under real_preserving preset when source_rig == target_rig
    res_noop = relight(scene, camera, src_rig, src_rig, EngineSettings.real_preserving(renderer_mode="ratio"))
    assert np.max(np.abs(res_noop.rgb_srgb - scene.rgb_srgb)) < 5e-4


def test_tier1_intrinsic_real_image_preservation_prevents_silhouette_collapse_and_matches_session():
    h, w = 48, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = np.where((xx > 12) & (xx < 40), 1.6, 4.0).astype(np.float32)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:, 12:26] = np.array([-0.55, 0.0, -0.83], dtype=np.float32)
    normals[:, 26:40] = np.array([0.85, 0.0, -0.52], dtype=np.float32)
    normals[:, :12] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    normals[:, 40:] = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    albedo_lin = np.full((h, w, 3), 0.45, dtype=np.float32)
    albedo_lin[:, 12:40] = np.array([0.06, 0.09, 0.18], dtype=np.float32)  # dark navy garment
    src_sh = np.full((h, w), 0.65, dtype=np.float32)
    rgb = linear_to_srgb(np.clip(albedo_lin * src_sh[..., None], 0.0, 1.0))

    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=depth,
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
        albedo=linear_to_srgb(albedo_lin),
        source_shading=src_sh,
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    src_rig = LightingRig(ambient_strength=0.45, lights=(
        Light("src", "directional", direction=(0.0, 0.0, -1.0), strength=0.6, casts_shadows=False),
    ))
    red_rig = LightingRig(ambient_color=(0.10, 0.03, 0.03), ambient_strength=0.05, lights=(
        Light("red_pt", "point", position=(-1.2, 0.0, 1.0), color=(1.0, 0.04, 0.06), strength=3.6, casts_shadows=False),
    ))

    res_old = relight(scene, camera, src_rig, red_rig, EngineSettings(renderer_mode="intrinsic"))
    cfg_real = EngineSettings.real_preserving(renderer_mode="intrinsic")
    res_new = relight(scene, camera, src_rig, red_rig, cfg_real)

    # Dark unlit right side of dark garment retains readability rather than crushing near zero
    old_unlit_mean = float(np.mean(res_old.rgb_srgb[:, 30:38]))
    new_unlit_mean = float(np.mean(res_new.rgb_srgb[:, 30:38]))
    assert new_unlit_mean > old_unlit_mean + 0.04

    # Distant background wall doesn't flood neon red
    old_bg_red = float(np.mean(res_old.rgb_srgb[:, 46:, 0] - res_old.rgb_srgb[:, 46:, 2]))
    new_bg_red = float(np.mean(res_new.rgb_srgb[:, 46:, 0] - res_new.rgb_srgb[:, 46:, 2]))
    assert new_bg_red < old_bg_red * 0.60

    # RelightSession matches relight() cleanly
    session = prepare_relight_session(scene, camera, src_rig, cfg_real)
    res_sess = session.render(red_rig)
    assert np.allclose(res_sess.rgb_srgb, res_new.rgb_srgb, atol=1e-5)


def test_tier1_highlight_relocation_attenuates_shadow_side_glare():
    h, w = 40, 64
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[:, 24:28] = 1.2
    rgb = np.full((h, w, 3), 0.32, dtype=np.float32)
    rgb[10:30, 40:54] = 0.90  # bright old highlight on right shadow side
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
    left_rig = LightingRig(ambient_strength=0.10, lights=(
        Light("left", "directional", direction=(-0.85, 0.0, -0.52), strength=2.2, casts_shadows=True),
    ))
    res_base = relight(scene, camera, src_rig, left_rig, EngineSettings(renderer_mode="ratio", base_blur_sigma=4.0))
    res_reloc = relight(
        scene, camera, src_rig, left_rig,
        EngineSettings(renderer_mode="ratio", base_blur_sigma=4.0, highlight_relocation_strength=0.9),
    )
    # Shadowed old highlight blob is suppressed further when highlight_relocation_strength is active
    assert float(np.mean(res_reloc.rgb_srgb[14:26, 42:52])) < float(np.mean(res_base.rgb_srgb[14:26, 42:52]))


def test_hybrid_neural_wavelet_mode_locks_multiscale_detail_and_masks():
    h, w = 32, 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.clip(0.45 + 0.09 * np.sin(xx * 1.6)[..., None], 0.0, 1.0).repeat(3, axis=-1).astype(np.float32)
    protected = np.zeros((h, w), dtype=np.float32)
    protected[:8, :] = 1.0
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=1.0 - protected,
        protected_mask=protected,
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig = LightingRig(ambient_strength=0.2, lights=(
        Light("tgt", "directional", strength=1.0, direction=(-0.5, 0.2, -0.8), casts_shadows=False),
    ))

    class SmoothRelighter:
        def render(self, sc, target_d, target_s, target_rig, settings):
            return NeuralRelightPrediction(rgb_srgb=np.full_like(sc.rgb_srgb, 0.74))

        def runtime_snapshot(self):
            return {"device": "cpu"}

        def close(self):
            return {"released": False}

    res = relight(
        scene,
        camera,
        rig,
        rig,
        EngineSettings(renderer_mode="hybrid_neural", detail_preservation_mode="wavelet", detail_lock_strength=1.0),
        neural_relighter=SmoothRelighter(),
    )
    assert np.allclose(res.rgb_srgb[:8, :, :], rgb[:8, :, :], atol=1e-5)
    patch = res.rgb_srgb[12:28, 8:40, 0]
    assert float(np.mean(patch)) > 0.60
    assert float(np.std(patch)) > 0.03


def test_tier2a_flux_iclight_adapter_protocol_registry_and_radiance_hint(monkeypatch):
    from relight_engine.adapters import FluxICLightRelighter, available_backends, create_relighter_backend
    from relight_engine.adapters.protocols import NeuralRelighter

    adapter = FluxICLightRelighter(provider="fal", api_key="test-key")
    assert isinstance(adapter, NeuralRelighter)
    assert "ic_light_v2" in available_backends()["relighters"]
    assert "flux_ic_light" in available_backends()["relighters"]
    assert isinstance(create_relighter_backend("ic_light_v2", provider="fal"), NeuralRelighter)

    h, w = 24, 32
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    rgb = np.full((h, w, 3), 0.45, dtype=np.float32)
    scene = SceneBuffers(
        rgb_srgb=rgb,
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
    )
    camera = CameraModel(60.0, 60.0, w / 2, h / 2)
    rig = LightingRig(ambient_strength=0.1, lights=(
        Light("left_red", "point", position=(-1.5, 0.2, 1.5), color=(1.0, 0.05, 0.08), strength=3.0, casts_shadows=False),
    ))
    prompt = adapter.describe_rig_prompt(scene, rig)
    assert "vivid red" in prompt
    assert "from the left" in prompt

    captured = {}

    def fake_run_fal(fg_srgb, hint_srgb, p, tw, th):
        captured["prompt"] = p
        captured["hint_shape"] = hint_srgb.shape
        return np.clip(hint_srgb * 0.7 + 0.15, 0.0, 1.0).astype(np.float32)

    monkeypatch.setattr(adapter, "_run_fal", fake_run_fal)
    res = relight(
        scene,
        camera,
        rig,
        rig,
        EngineSettings(renderer_mode="hybrid_neural", detail_preservation_mode="wavelet"),
        neural_relighter=adapter,
    )
    assert res.rgb_srgb.shape == (h, w, 3)
    assert res.metrics["renderer_mode"] == "hybrid_neural"
    assert res.metrics["neural_relighter"]["backend"] == "ic_light_v2_flux"
    assert res.metrics["neural_relighter"]["radiance_hint_used"] is True
    assert "from the left" in captured["prompt"]

    no_key = FluxICLightRelighter(provider="fal", api_key=None)
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FAL_KEY"):
        relight(scene, camera, rig, rig, EngineSettings(renderer_mode="hybrid_neural"), neural_relighter=no_key)
