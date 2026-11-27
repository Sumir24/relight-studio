from __future__ import annotations
from collections import OrderedDict
import hashlib
import threading
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from .color import srgb_to_linear, linear_to_srgb, luminance
from .contracts import SceneBuffers, EngineSettings, LightingRig
from .wavelet import compose_wavelet_identity_lock

_INTRINSIC_DETAIL_CACHE: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
_INTRINSIC_RESIDUAL_CACHE: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
_INTRINSIC_DETAIL_LOCK = threading.Lock()
_INTRINSIC_CACHE_MAX = 4


def _array_digest(arr: np.ndarray | None) -> tuple | None:
    if arr is None:
        return None
    buf = np.ascontiguousarray(arr)
    return (buf.shape, str(buf.dtype), hashlib.blake2b(memoryview(buf), digest_size=16).digest())


def _filter_log_field(field: np.ndarray, sigma: float, settings: EngineSettings, *, eps_guide: float) -> np.ndarray:
    if settings.edge_aware_base:
        return _edge_aware_log_base(field, sigma, eps_guide=eps_guide)
    return gaussian_filter(field, sigma, mode='reflect')

def _edge_aware_log_base(log_y: np.ndarray, sigma: float, eps_guide: float = 0.08) -> np.ndarray:
    """Self-guided filter in log-luminance space to suppress boundary halos."""
    mean_i = gaussian_filter(log_y, sigma, mode='reflect')
    mean_ii = gaussian_filter(log_y * log_y, sigma, mode='reflect')
    var_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)
    a = var_i / (var_i + eps_guide)
    b = (1.0 - a) * mean_i
    mean_a = gaussian_filter(a, sigma, mode='reflect')
    mean_b = gaussian_filter(b, sigma, mode='reflect')
    return mean_a * log_y + mean_b


def _compute_log_base(log_y: np.ndarray, settings: EngineSettings) -> np.ndarray:
    if settings.edge_aware_base:
        return _edge_aware_log_base(log_y, settings.base_blur_sigma)
    return gaussian_filter(log_y, settings.base_blur_sigma, mode='reflect')


def _joint_guided_filter(field: np.ndarray, guide: np.ndarray, sigma: float, eps_guide: float = 0.005) -> np.ndarray:
    """Edge-preserving guided filter using a structural guide (e.g. log-luminance) to prevent chroma halos."""
    mean_i = gaussian_filter(guide, sigma, mode='reflect')
    mean_ii = gaussian_filter(guide * guide, sigma, mode='reflect')
    var_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)
    mean_p = gaussian_filter(field, sigma, mode='reflect')
    mean_ip = gaussian_filter(guide * field, sigma, mode='reflect')
    cov_ip = mean_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps_guide)
    b = mean_p - a * mean_i
    return gaussian_filter(a, sigma, mode='reflect') * guide + gaussian_filter(b, sigma, mode='reflect')


def _preserve_source_chroma(
    src_lin: np.ndarray,
    albedo_lin: np.ndarray,
    settings: EngineSettings,
    eps: float = 1e-4,
) -> np.ndarray:
    """Restore high-frequency local chromaticity missing from smooth albedo without copying broad source casts."""
    strength = 0.75 * float(settings.detail_lock_strength)
    if strength <= 0.0:
        return albedo_lin

    y_src = np.maximum(luminance(src_lin), eps)
    y_alb = np.maximum(luminance(albedo_lin), eps)
    log_y_src = np.log(y_src)
    log_c_src = np.log(np.maximum(src_lin, eps)) - log_y_src[..., None]
    log_c_alb = np.log(np.maximum(albedo_lin, eps)) - np.log(y_alb)[..., None]
    c_diff = log_c_src - log_c_alb

    # Separate broad source color casts (low-frequency) from local fine chromatic detail (high-frequency).
    # Joint filtering guided by log_y_src preserves structural edges (no halos) while capturing isoluminant chroma detail.
    sigma = float(settings.base_blur_sigma)
    if settings.edge_aware_base:
        base_c = np.stack([
            _joint_guided_filter(c_diff[..., c], log_y_src, sigma, eps_guide=0.005)
            for c in range(3)
        ], axis=-1)
    else:
        base_c = np.stack([
            gaussian_filter(c_diff[..., c], sigma, mode='reflect')
            for c in range(3)
        ], axis=-1)
    delta_c_hf = np.clip(c_diff - base_c, -0.65, 0.65)

    # Attenuate chroma restoration in near-black pixels where RGB ratios are noise-dominated.
    chroma_conf = np.clip((np.minimum(y_src, y_alb) - 0.01) / 0.06, 0.0, 1.0)
    corrected = albedo_lin * np.exp(delta_c_hf * (strength * chroma_conf)[..., None])
    # Re-normalize to keep predicted albedo luminance unchanged so target lighting drives shading.
    corrected *= (y_alb / np.maximum(luminance(corrected), eps))[..., None]
    return np.clip(corrected, 0.0, 1.0).astype(np.float32)


def _depth_guided_smooth(
    field: np.ndarray, depth: np.ndarray, sigma: float, eps_guide: float = 0.008
) -> np.ndarray:
    """Convex edge-stopping bilateral smooth preserving sharp depth silhouettes without halo overshoot."""
    if sigma <= 0.0:
        return field
    d_f32 = np.asarray(depth, dtype=np.float32)
    valid = d_f32 > 0
    guide = np.log(np.maximum(d_f32, 1e-3)).astype(np.float32)
    # Use a 2-pass separable-style 9-tap + 13-tap convex bilateral stencil scaled by sigma
    # so every smoothed value stays strictly inside local [min, max] bounds (zero linear overshoot).
    rad_outer = max(1, int(round(sigma * 1.4)))
    rad_inner = max(1, int(round(sigma * 0.7)))
    offsets = [
        (-rad_inner, 0, 1.0), (rad_inner, 0, 1.0), (0, -rad_inner, 1.0), (0, rad_inner, 1.0),
        (-rad_inner, -rad_inner, 0.7), (-rad_inner, rad_inner, 0.7), (rad_inner, -rad_inner, 0.7), (rad_inner, rad_inner, 0.7),
        (-rad_outer, 0, 0.55), (rad_outer, 0, 0.55), (0, -rad_outer, 0.55), (0, rad_outer, 0.55),
    ]
    inv_2sig_d = np.float32(-0.5 / max(float(eps_guide) * 8.0, 0.0036))

    def _bilateral_2d(p: np.ndarray) -> np.ndarray:
        p32 = p.astype(np.float32, copy=False)
        h, w = p32.shape
        acc = p32.copy()
        wsum = np.ones((h, w), dtype=np.float32)
        for dy, dx, spatial_w in offsets:
            pn = np.roll(np.roll(p32, dy, axis=0), dx, axis=1)
            gn = np.roll(np.roll(guide, dy, axis=0), dx, axis=1)
            vn = np.roll(np.roll(valid, dy, axis=0), dx, axis=1).copy()
            if dy > 0:
                vn[:dy, :] = False
            elif dy < 0:
                vn[dy:, :] = False
            if dx > 0:
                vn[:, :dx] = False
            elif dx < 0:
                vn[:, dx:] = False
            dg = guide - gn
            w_n = np.where(valid & vn, np.float32(spatial_w) * np.exp(np.maximum(dg * dg * inv_2sig_d, -20.0)), np.float32(0.0))
            acc += w_n * pn
            wsum += w_n
        return (acc / np.maximum(wsum, np.float32(1e-6))).astype(np.float32)

    if field.ndim == 2:
        return _bilateral_2d(field)
    channels = [_bilateral_2d(field[..., c]) for c in range(field.shape[-1])]
    return np.stack(channels, axis=-1).astype(np.float32)


def _active_blend_mask(
    scene: SceneBuffers, geometry_confidence: np.ndarray, settings: EngineSettings
) -> np.ndarray:
    mask = np.clip(scene.lightable_mask, 0, 1).astype(np.float32, copy=True)
    if scene.protected_mask is not None:
        mask *= 1.0 - np.clip(scene.protected_mask, 0, 1)
    if scene.alpha_mask is not None:
        mask *= np.clip(scene.alpha_mask, 0, 1)
    if getattr(settings, "protect_sky", True):
        from .scene_policy import detect_sky_horizon_mask
        sky_w = detect_sky_horizon_mask(scene)
        if np.any(sky_w > 1e-4):
            mask *= 1.0 - sky_w
    conf = np.clip(geometry_confidence, 0, 1).astype(np.float32)
    min_c = float(settings.min_geometry_confidence)
    ramp_hi = max(min_c + 0.15, 0.30)
    if ramp_hi > min_c + 1e-6:
        conf_weight = np.clip((conf - min_c) / (ramp_hi - min_c), 0.0, 1.0)
    else:
        conf_weight = (conf >= min_c).astype(np.float32)
    mask *= conf_weight * (conf >= min_c).astype(np.float32)
    return np.clip(mask, 0, 1).astype(np.float32)


def _apply_highlight_shoulder(
    relit: np.ndarray,
    src: np.ndarray,
    eps: float = 1e-4,
    *,
    filmic: bool = False,
) -> np.ndarray:
    peak = np.max(relit, axis=-1)
    source_peak = np.max(src, axis=-1)
    if filmic:
        threshold, width = 0.80, 0.12
        over = np.maximum(peak - threshold, 0.0)
        compressed = threshold + width * np.tanh(over / width)
        newly_hot = (peak > threshold) & (peak > source_peak + 1e-6)
        scale = np.where(newly_hot, compressed / np.maximum(peak, eps), 1.0)
        scaled = relit * scale[..., None]
        # Perceptual highlight desaturation toward neutral luminance when overlapping lights saturate.
        luma = np.maximum(luminance(scaled), eps)[..., None]
        desat = np.where(newly_hot, np.clip(over * 0.55, 0.0, 0.65), 0.0)[..., None]
        return (scaled * (1.0 - desat) + luma * desat).astype(np.float32)
    threshold, width = 0.82, 0.10
    compressed = threshold + width * (1 - np.exp(-np.maximum(peak - threshold, 0) / width))
    newly_hot = (peak > threshold) & (peak > source_peak + 1e-6)
    scale = np.where(newly_hot, compressed / np.maximum(peak, eps), 1.0)
    return relit * scale[..., None]


def _infer_atmosphere_color(target_rig: LightingRig | None, settings: EngineSettings) -> np.ndarray:
    if settings.atmosphere_color is not None:
        return np.asarray(settings.atmosphere_color, dtype=np.float32)
    if target_rig is not None:
        weighted = np.zeros(3, dtype=np.float32)
        total_w = 0.0
        for light in target_rig.lights:
            w = max(float(light.strength), 0.0)
            weighted += np.asarray(light.color, dtype=np.float32) * w
            total_w += w
        if target_rig.ambient_strength > 0.0:
            w_amb = float(target_rig.ambient_strength) * 0.5
            weighted += np.asarray(target_rig.ambient_color, dtype=np.float32) * w_amb
            total_w += w_amb
        if total_w > 1e-6:
            col = weighted / total_w
            return col / max(float(luminance(col)), 1e-4)
    return np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _apply_aerial_perspective(
    relit: np.ndarray,
    scene: SceneBuffers,
    target_rig: LightingRig | None,
    settings: EngineSettings,
) -> np.ndarray:
    density = float(settings.atmosphere_density)
    if density <= 0.0:
        return relit
    valid = scene.depth > 0
    if not np.any(valid):
        return relit
    z_near, z_far = np.quantile(scene.depth[valid], [0.02, 0.98])
    z_span = max(float(z_far - z_near), 1e-4)
    d_norm = np.clip((scene.depth - float(z_near)) / z_span, 0.0, 1.25)
    trans = np.where(valid, np.exp(-density * d_norm), 1.0).astype(np.float32)

    relit_y = np.maximum(luminance(relit), 1e-4)
    ref_y = float(np.quantile(relit_y[valid], 0.65))
    scatter_tint = _infer_atmosphere_color(target_rig, settings)[None, None, :]
    haze_rad = scatter_tint * (0.55 * relit_y[..., None] + 0.45 * ref_y)
    return (relit * trans[..., None] + haze_rad * (1.0 - trans[..., None])).astype(np.float32)


def _estimate_background_weight(scene: SceneBuffers) -> np.ndarray:
    """Domain-agnostic soft background weight in [0, 1] from explicit masks or relative depth."""
    h, w = scene.shape
    if "background" in scene.semantic_masks:
        bg = np.clip(scene.semantic_masks["background"], 0.0, 1.0).astype(np.float32)
        return np.clip(_depth_guided_smooth(bg, scene.depth, 3.0), 0.0, 1.0)

    depth = scene.depth.astype(np.float32)
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((h, w), dtype=np.float32)

    z_vals = depth[valid]
    q10, q90 = np.quantile(z_vals, [0.10, 0.90])
    if float(q90 - q10) < 1e-3:
        if scene.semantic_masks:
            fg = np.zeros((h, w), dtype=np.float32)
            for m in scene.semantic_masks.values():
                fg = np.maximum(fg, np.clip(m, 0.0, 1.0))
            return np.clip(1.0 - fg, 0.0, 1.0)
        return np.zeros((h, w), dtype=np.float32)

    if scene.semantic_masks:
        fg = np.zeros((h, w), dtype=np.float32)
        for m in scene.semantic_masks.values():
            fg = np.maximum(fg, np.clip(m, 0.0, 1.0))
        bg_mask = np.clip(1.0 - fg, 0.0, 1.0)
        fg_pixels = valid & (fg > 0.5)
        if np.any(fg_pixels):
            z_fg = float(np.median(depth[fg_pixels]))
            z_hi = max(float(q90), z_fg + 0.15 * float(q90 - q10))
            depth_gate = np.clip((depth - z_fg) / max(z_hi - z_fg, 1e-3), 0.0, 1.0)
            bg = bg_mask * (0.78 + 0.22 * depth_gate)
        else:
            bg = bg_mask
    else:
        z_lo = float(q10) + 0.45 * float(q90 - q10)
        z_hi = max(float(q90), z_lo + 1e-3)
        bg = np.clip((depth - z_lo) / (z_hi - z_lo), 0.0, 1.0)

    return np.clip(_depth_guided_smooth(bg.astype(np.float32), depth, 3.0), 0.0, 1.0)


def _attenuate_background_lighting(
    scene: SceneBuffers,
    target_diffuse: np.ndarray,
    target_specular: np.ndarray,
    settings: EngineSettings,
    *,
    source_diffuse: np.ndarray | None = None,
    albedo_lin: np.ndarray | None = None,
    src_lin: np.ndarray | None = None,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Attenuate excessive target-light flooding on excluded regions while preserving physical room walls/floors."""
    bg_inf = float(settings.background_light_influence)
    env_inf = float(getattr(settings, "environment_light_influence", bg_inf))
    if bg_inf >= 1.0 - 1e-6 and env_inf >= 1.0 - 1e-6:
        return target_diffuse, target_specular

    from .scene_policy import classify_scene_layers
    _, env_surf, _ = classify_scene_layers(scene)
    bg_w = _estimate_background_weight(scene)
    local_inf = (1.0 - env_surf) * bg_inf + env_surf * env_inf
    atten = np.clip(bg_w * (1.0 - local_inf), 0.0, 1.0)[..., None]
    if not np.any(atten > 1e-5):
        return target_diffuse, target_specular

    if source_diffuse is not None:
        base_d = np.maximum(source_diffuse, 0.0)
    elif albedo_lin is not None and src_lin is not None:
        implied = np.clip(src_lin / np.maximum(albedo_lin, 0.05), 0.05, 2.5)
        base_d = np.maximum(_depth_guided_smooth(implied, scene.depth, 4.0), eps)
    else:
        base_d = np.maximum(target_diffuse, 0.0)

    t_d_eff = np.maximum(target_diffuse, 0.0) * (1.0 - atten) + base_d * atten
    t_s_eff = np.maximum(target_specular, 0.0) * (1.0 - atten)
    return t_d_eff.astype(np.float32), t_s_eff.astype(np.float32)


def _soft_knee_bright_ratio(
    ratio: np.ndarray,
    rgb_gain: np.ndarray,
    y_src: np.ndarray,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothly compress gain above 1.0 only on genuinely high-luminance pixels to prevent blowout."""
    bright_gate = np.clip((y_src - 0.34) / 0.32, 0.0, 1.0)
    if not np.any(bright_gate > 1e-4) or not np.any(ratio > 1.0):
        return ratio, rgb_gain
    max_excess = np.maximum(0.88 / np.maximum(y_src, 0.35) - 1.0, 0.10)
    excess = np.maximum(ratio - 1.0, 0.0)
    damped_excess = max_excess * np.tanh(excess / max_excess)
    soft_ratio = np.where(
        ratio > 1.0,
        1.0 + (1.0 - 0.85 * bright_gate) * excess + (0.85 * bright_gate) * damped_excess,
        ratio,
    ).astype(np.float32)
    scale = (soft_ratio / np.maximum(ratio, eps))[..., None]
    return soft_ratio, (rgb_gain * scale).astype(np.float32)


def _apply_perceptual_color_anchor(
    src_lin: np.ndarray,
    y_src: np.ndarray,
    s_d: np.ndarray,
    t_d: np.ndarray,
    ratio: np.ndarray,
    rgb_gain: np.ndarray,
    conf_norm: np.ndarray,
    change_w: np.ndarray,
    settings: EngineSettings,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute perceptual log-luminance & linear-chroma anchored illumination delta.

    Prevents dark saturated materials (e.g. navy suits, hair) and human skin from turning
    purple/black under saturated colored lights while allowing strong lit-side response.
    """
    real_pres = float(settings.real_image_preservation)
    c_strength = float(settings.color_transfer_strength)
    s_y = np.maximum(luminance(s_d), eps)
    t_y = np.maximum(luminance(t_d), eps)

    # Measure direct light activity by peak channel as well as luminance so saturated
    # red/blue lights aren't misclassified as shadows due to low Rec.709 luminance weight.
    s_peak = np.maximum(np.max(s_d, axis=-1), eps)
    t_peak = np.maximum(np.max(t_d, axis=-1), eps)
    peak_ratio = np.clip(t_peak / s_peak, 0.0, settings.ratio_max)

    if real_pres <= 0.0:
        if c_strength < 1.0 - 1e-6:
            gain_y = np.maximum(luminance(rgb_gain), eps)[..., None]
            chroma_shift = rgb_gain / gain_y
            hi_desat = np.clip((y_src - 0.50) / 0.45, 0.0, 1.0)[..., None] * 0.45
            lit_gate = np.clip(t_y / np.maximum(s_y * 0.85, 0.25), 0.15, 1.0)[..., None]
            eff_c = c_strength * (1.0 - hi_desat) * lit_gate
            chroma_eff = 1.0 + eff_c * (chroma_shift - 1.0)
            rgb_gain = ratio[..., None] * chroma_eff
        return ratio.astype(np.float32), rgb_gain.astype(np.float32)

    # 1. Soften downward luminance collapse on shadow sides while preserving brightening (>= 1.0).
    # Dark pixels (e.g. navy fabric, dark hair) and fill-preserved scenes avoid unintended black crushing.
    dark_sens = np.clip((0.15 - y_src) / 0.12, 0.0, 0.50)
    fill_bal = float(np.clip(getattr(settings, "preserve_fill_balance", 0.0), 0.0, 1.0))
    shadow_preserve = np.clip(0.55 * real_pres + dark_sens * real_pres + 0.35 * fill_bal * (1.0 - real_pres), 0.0, 0.88)
    # Lit saturated colors get partial perceptual brightness support from their peak channel.
    perceptual_ratio = np.maximum(ratio, 0.55 * peak_ratio + 0.45 * ratio)
    effective_ratio = np.where(
        perceptual_ratio < 1.0,
        1.0 - (1.0 - shadow_preserve) * (1.0 - perceptual_ratio),
        perceptual_ratio,
    )
    # Bound log-luminance swing gracefully in real-preserving / fill-balanced mode.
    log_r = np.log(np.maximum(effective_ratio, eps))
    min_ratio_floor = max(0.40 + 0.40 * real_pres + 0.16 * fill_bal, 0.25)
    max_darken_log = float(np.log(min(min_ratio_floor, 0.92)))
    effective_ratio = np.exp(np.clip(log_r, max_darken_log, np.log(settings.ratio_max))).astype(np.float32)

    # 2. Anchor output chromaticity to source linear chroma weighted by lit confidence and dark gate.
    # Chroma blending in linear unit-luminance space prevents extreme R/B multipliers (e.g. 3.9x)
    # from mutating navy -> purple or skin -> neon mask.
    y_safe = np.maximum(y_src, eps)[..., None]
    chroma_src = src_lin / y_safe
    raw_tgt = src_lin * rgb_gain
    raw_tgt_y = np.maximum(luminance(raw_tgt), eps)[..., None]
    chroma_tgt = raw_tgt / raw_tgt_y

    lit_conf = np.clip((peak_ratio - 0.15) / 0.65, 0.0, 1.0)
    dark_chroma_gate = np.clip((y_src - 0.008) / 0.05, 0.22, 1.0)
    hi_desat = np.clip((y_src - 0.55) / 0.40, 0.0, 1.0) * 0.45
    c_blend_base = np.clip(c_strength * lit_conf * dark_chroma_gate * (1.0 - hi_desat) * conf_norm, 0.0, 0.85)
    # Neutral studio walls/floors (low intrinsic saturation) accept colored light gels cleanly while
    # saturated skin/textiles retain tighter perceptual albedo fidelity.
    if float(getattr(settings, "environment_light_influence", 1.0)) >= 0.70 and float(settings.background_light_influence) >= 0.60:
        src_max = np.maximum(np.max(src_lin, axis=-1), eps)
        src_sat = 1.0 - np.min(src_lin, axis=-1) / src_max
        neutral_wall_w = np.clip((0.18 - src_sat) / 0.18, 0.0, 1.0)
        c_blend_base = np.clip(c_blend_base + 0.28 * c_strength * lit_conf * neutral_wall_w * conf_norm, 0.0, 0.92)
    c_blend = c_blend_base[..., None]
    chroma_out = chroma_src * (1.0 - c_blend) + chroma_tgt * c_blend
    chroma_out_y = np.maximum(luminance(chroma_out), eps)[..., None]
    chroma_out = chroma_out / chroma_out_y

    anchored_rgb_gain = np.maximum((chroma_out / np.maximum(chroma_src, eps)) * effective_ratio[..., None], eps)
    return effective_ratio.astype(np.float32), anchored_rgb_gain.astype(np.float32)


def _apply_regional_protection_and_sheen(
    scene: SceneBuffers,
    src_lin: np.ndarray,
    relit_lin: np.ndarray,
    y_src: np.ndarray,
    s_d: np.ndarray,
    t_d_eff: np.ndarray,
    spec_delta: np.ndarray,
    conf_norm: np.ndarray,
    change_w: np.ndarray,
    settings: EngineSettings,
    eps: float = 1e-4,
) -> np.ndarray:
    """Apply regional identity protection (background + dark materials) and directional surface sheen."""
    real_pres = float(settings.real_image_preservation)
    out = relit_lin

    if real_pres > 0.0 and np.any(change_w > 1e-6):
        from .scene_policy import classify_scene_layers
        _, env_surf, _ = classify_scene_layers(scene)
        bg_w = _estimate_background_weight(scene)
        local_bg_inf = (1.0 - env_surf) * float(settings.background_light_influence) + env_surf * float(getattr(settings, "environment_light_influence", settings.background_light_influence))
        bg_prot = bg_w * (1.0 - local_bg_inf) * (0.45 + 0.55 * real_pres)
        dark_prot = np.clip((0.15 - y_src) / 0.12, 0.0, 0.45) * real_pres
        # Protect unlit/shadow regions more strongly than actively light-facing foreground regions.
        t_peak = np.max(t_d_eff, axis=-1)
        s_peak = np.maximum(np.max(s_d, axis=-1), eps)
        lit_rel = np.clip(t_peak / s_peak, 0.0, 1.0)
        fg_prot = dark_prot * (1.0 - 0.35 * lit_rel)
        protection = np.clip(np.maximum(bg_prot, fg_prot), 0.0, 0.90) * conf_norm
        out = src_lin * protection[..., None] + out * (1.0 - protection[..., None])

        # Directional dielectric/microfiber sheen lets colored/key lights visibly graze dark
        # garments and skin contours on light-facing surfaces without staining the whole garment.
        bg_atten = 1.0 - bg_w * (1.0 - float(settings.background_light_influence))
        t_norm = t_d_eff / np.maximum(s_peak[..., None], 0.5)
        lit_excess = np.clip(t_peak / np.maximum(s_peak * 0.45, 0.20) - 0.25, 0.0, 1.0)
        sheen_drive = np.clip(t_peak, 0.0, 1.5) * lit_excess
        smooth_sigma = max(float(settings.lighting_smooth_sigma) * 2.5, 3.0)
        sheen_drive = np.maximum(_depth_guided_smooth(sheen_drive, scene.depth, smooth_sigma), 0.0)
        c_str = float(settings.color_transfer_strength)
        sheen_col = t_norm / np.maximum(luminance(t_norm), eps)[..., None]
        sheen_col = 1.0 + np.clip(c_str, 0.25, 1.0) * (sheen_col - 1.0)
        sheen_col = sheen_col / np.maximum(np.max(sheen_col, axis=-1, keepdims=True), 1.0)
        sheen_tone = np.clip((y_src - 0.002) / 0.012, 0.25, 1.0)
        sheen_energy = (
            sheen_drive
            * bg_atten
            * conf_norm
            * change_w
            * sheen_tone
            * (0.042 * (1.0 - 0.55 * bg_w))
        )
        out = out + sheen_col * sheen_energy[..., None]

    # Add incremental virtual-light GGX highlight, gated by geometry confidence and void-dust guard on priors.
    if scene.metalness is None and scene.specular_level is None and scene.clearcoat is None:
        void_guard = np.clip((y_src - 0.003) / 0.020, 0.0, 1.0)[..., None]
        bright_spec_guard = np.clip((0.86 - np.max(out, axis=-1)) / 0.30, 0.12, 1.0)[..., None]
        spec_delta = spec_delta * void_guard * bright_spec_guard
    reloc_s = float(np.clip(settings.highlight_relocation_strength, 0.0, 2.0))
    if reloc_s > 0.0:
        spec_delta = spec_delta * (1.0 + 0.75 * reloc_s)
    scaled_spec = np.maximum(spec_delta, 0.0) * np.maximum(settings.specular_scale, 0.0)
    max_spec_cap = np.maximum(out * (1.5 + 1.0 * reloc_s) + 0.45 * np.maximum(settings.specular_scale, 0.0), eps)
    bounded_spec = max_spec_cap * np.tanh(scaled_spec / max_spec_cap)
    out = out + bounded_spec
    return np.maximum(out, 0.0).astype(np.float32)


def _compute_detail_lock_weight_map(
    scene: SceneBuffers,
    geometry_confidence: np.ndarray,
    ratio: np.ndarray,
    residual_layers: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    settings: EngineSettings,
    eps: float = 1e-4,
) -> np.ndarray:
    """Distinguish surface texture from confident macro illumination/artifact edges.

    Lowers detail copying across strong macro source-shading boundaries and confident broad glare
    lobes when relighting shifts illumination, while keeping genuine albedo/texture edges locked.
    """
    h, w = scene.shape
    min_c = float(settings.min_geometry_confidence)
    geom_w = np.clip((geometry_confidence - min_c) / 0.30, 0.0, 1.0).astype(np.float32)
    change_w = np.clip(np.abs(ratio - 1.0) / 0.25, 0.0, 1.0).astype(np.float32)

    if scene.source_shading is not None:
        sh_gray = scene.source_shading if scene.source_shading.ndim == 2 else luminance(scene.source_shading)
        log_sh = gaussian_filter(np.log(np.maximum(sh_gray, eps)).astype(np.float32), 2.5, mode="reflect")
        g_sh = np.hypot(sobel(log_sh, axis=1, mode="reflect") / 8.0, sobel(log_sh, axis=0, mode="reflect") / 8.0)
    else:
        g_sh = np.zeros((h, w), dtype=np.float32)

    if scene.albedo is not None:
        y_alb = np.maximum(luminance(srgb_to_linear(scene.albedo)), eps)
        log_alb = np.log(y_alb).astype(np.float32)
        g_alb = np.hypot(sobel(log_alb, axis=1, mode="reflect") / 8.0, sobel(log_alb, axis=0, mode="reflect") / 8.0)
    else:
        g_alb = np.zeros((h, w), dtype=np.float32)

    illum_edge = np.clip((g_sh - 0.5 * g_alb - 0.015) / 0.06, 0.0, 1.0).astype(np.float32)
    if residual_layers is not None:
        _, _, res_lobe = residual_layers
        # Protect crisp textured/albedo features (e.g., dial typography & brushed bezel grain)
        # from broad glare detail-lock suppression.
        edge_protect = np.clip(g_alb / 0.06, 0.0, 0.85).astype(np.float32)
        glare_w = (np.clip(np.max(res_lobe, axis=-1) / 0.08, 0.0, 1.0) * (1.0 - edge_protect)).astype(np.float32)
    else:
        glare_w = np.zeros((h, w), dtype=np.float32)

    artifact_suppress = np.maximum(illum_edge * change_w, glare_w) * geom_w
    return np.clip(1.0 - 0.70 * artifact_suppress, 0.25, 1.0).astype(np.float32)


def _finalize_protected_display_output(
    scene: SceneBuffers,
    relit_lin: np.ndarray,
    src_lin: np.ndarray,
    mask: np.ndarray,
    settings: EngineSettings,
    eps: float = 1e-4,
) -> np.ndarray:
    """Apply display highlight rolloff only to active pixels and guarantee bit-exact protected regions."""
    relit_blended = src_lin * (1.0 - mask[..., None]) + relit_lin * mask[..., None]
    shouldered = _apply_highlight_shoulder(
        relit_blended,
        src_lin,
        eps,
        filmic=bool(settings.filmic_highlight_rolloff),
    )
    # Re-blend with mask so display shoulder never perturbs partially or fully protected pixels.
    relit_safe = src_lin * (1.0 - mask[..., None]) + shouldered * mask[..., None]
    out = linear_to_srgb(relit_safe)
    exact_prot = mask <= 1e-7
    if np.any(exact_prot):
        out = np.where(exact_prot[..., None], scene.rgb_srgb, out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compose_source_preserving(scene: SceneBuffers, source_diffuse: np.ndarray,
                              target_diffuse: np.ndarray, source_specular: np.ndarray,
                              target_specular: np.ndarray, geometry_confidence: np.ndarray,
                              settings: EngineSettings,
                              target_rig: LightingRig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Transfer lighting in source-preserving fallback mode with multiscale highlight/shadow separation."""
    eps = 1e-4
    if (
        not settings.preserve_source_lighting
        and abs(float(settings.output_exposure) - 1.0) < 1e-6
        and float(settings.atmosphere_density) <= 0.0
        and np.allclose(source_diffuse, target_diffuse, atol=1e-6)
        and np.allclose(source_specular, target_specular, atol=1e-6)
    ):
        return np.clip(scene.rgb_srgb, 0.0, 1.0).astype(np.float32), np.ones(scene.shape, dtype=np.float32)

    src = srgb_to_linear(scene.rgb_srgb)
    t_d_in, t_s_in = _attenuate_background_lighting(
        scene,
        target_diffuse,
        target_specular,
        settings,
        source_diffuse=source_diffuse,
        src_lin=src,
        eps=eps,
    )
    s_d = np.maximum(source_diffuse, eps)
    t_d = np.maximum(t_d_in, 0.0)
    s_y = np.maximum(luminance(s_d), eps)
    t_y = np.maximum(luminance(t_d), eps)
    ratio = np.clip(t_y / s_y, settings.ratio_min, settings.ratio_max)

    min_gain = settings.ratio_min * settings.color_ratio_min
    max_gain = settings.ratio_max * settings.color_ratio_max
    # Maintain ratio_min as the shadow floor for low-saturation lights (e.g. outdoor daylight),
    # while allowing saturated colored lights in deep shadow to attenuate toward min_gain without red/color wash.
    t_chan_max = np.maximum(np.max(t_d, axis=-1, keepdims=True), eps)
    t_hue = t_d / t_chan_max
    t_sat = 1.0 - np.min(t_hue, axis=-1, keepdims=True)
    sat_w = np.clip((t_sat - 0.25) / 0.55, 0.0, 1.0)
    dyn_min_gain = float(settings.ratio_min) * (1.0 - sat_w) + min_gain * sat_w
    rgb_gain = np.clip(t_d / s_d, dyn_min_gain, max_gain)
    if settings.lighting_smooth_sigma > 0.0:
        ratio = np.clip(
            _depth_guided_smooth(ratio, scene.depth, settings.lighting_smooth_sigma),
            settings.ratio_min, settings.ratio_max,
        )
        rgb_gain = np.clip(
            _depth_guided_smooth(rgb_gain, scene.depth, settings.lighting_smooth_sigma),
            min_gain, max_gain,
        )

    y = np.maximum(luminance(src), eps)
    ratio, rgb_gain = _soft_knee_bright_ratio(ratio, rgb_gain, y, eps=eps)
    log_y = np.log(y)
    base = np.exp(_compute_log_base(log_y, settings))
    detail = y - base

    change_w = np.clip(
        np.maximum(np.abs(ratio - 1.0), np.linalg.norm(rgb_gain - 1.0, axis=-1) * 0.45),
        0.0, 1.0,
    )
    min_c = float(settings.min_geometry_confidence)
    conf_norm = np.clip((geometry_confidence - min_c) / 0.35, 0.0, 1.0).astype(np.float32)

    reloc_s = float(settings.highlight_relocation_strength)
    if settings.preserve_source_lighting or not np.any(change_w > 1e-6):
        base_eff = base
        rgb_gain_eff = rgb_gain
    else:
        # Only compress isolated specular overshoot (> 1.35x macro albedo above 0.45 linear) or
        # extreme near-clipped plateaus (> 0.80 linear), leaving snow, sky, and white fabric untouched.
        sigma_macro = float(settings.base_blur_sigma) * 2.5
        base_macro = np.exp(_filter_log_field(log_y, sigma_macro, settings, eps_guide=0.12))
        spec_thresh = np.minimum(np.maximum(base_macro * 1.35, 0.45), 0.80)
        excess = np.maximum(base - spec_thresh, 0.0)
        base_sep = base - excess * (0.55 * change_w * conf_norm)

        if reloc_s > 0.0:
            # Tier 1D: Strongly attenuate baked specular lobes facing away from the new light.
            reloc_thresh = np.minimum(np.maximum(base_macro * 1.15, 0.32), 0.70)
            reloc_excess = np.maximum(base - reloc_thresh, 0.0)
            away_from_new = np.clip((1.05 - ratio) / 0.75, 0.0, 1.0)
            extra_suppress = reloc_excess * (0.75 * reloc_s * away_from_new * change_w * conf_norm)
            base_sep = np.maximum(base_sep - extra_suppress, eps)

        base_eff = base_sep

        # Prevent near-white baked source highlights from picking up >1.0 saturated channel amplification.
        white_hi = np.clip((y - 0.58) / 0.35, 0.0, 1.0) * change_w * conf_norm
        gain_over = np.maximum(rgb_gain - 1.0, 0.0)
        rgb_gain_eff = rgb_gain - gain_over * (0.55 * white_hi[..., None])

    ratio, rgb_gain_eff = _apply_perceptual_color_anchor(
        src,
        y,
        s_d,
        t_d,
        ratio,
        rgb_gain_eff,
        conf_norm,
        change_w,
        settings,
        eps=eps,
    )

    y_sep = np.maximum(base_eff + detail, eps)

    tinted = src * (y_sep / y)[..., None] * rgb_gain_eff
    eff_y = np.maximum(luminance(tinted), eps)
    base_ratio = eff_y / np.maximum(y_sep, eps)
    if abs(float(settings.detail_power) - 1.0) < 1e-6:
        y_relit = np.maximum(y_sep * base_ratio, eps)
    else:
        y_relit = np.maximum(
            base_eff * base_ratio + detail * np.power(np.clip(base_ratio, settings.ratio_min, settings.ratio_max), settings.detail_power),
            eps,
        )
    relit = tinted * (y_relit / eff_y)[..., None]
    relit *= settings.output_exposure

    spec_delta = np.maximum(t_s_in - source_specular, 0.0) * conf_norm[..., None]
    relit = _apply_regional_protection_and_sheen(
        scene,
        src,
        relit,
        y,
        s_d,
        t_d,
        spec_delta,
        conf_norm,
        change_w,
        settings,
        eps=eps,
    )
    relit = _apply_aerial_perspective(relit, scene, target_rig, settings)

    if settings.detail_preservation_mode == "wavelet":
        detail_w = _compute_detail_lock_weight_map(
            scene, geometry_confidence, ratio, None, settings, eps=eps
        )
        relit = compose_wavelet_identity_lock(
            src,
            relit,
            levels=settings.wavelet_levels,
            detail_lock_strength=settings.detail_lock_strength,
            mid_blend=settings.wavelet_mid_blend,
            detail_weight_map=detail_w,
            eps=eps,
        )

    mask = _active_blend_mask(scene, geometry_confidence, settings)
    out = _finalize_protected_display_output(scene, relit, src, mask, settings, eps=eps)
    return out, ratio.astype(np.float32)


def _compute_f3_log_detail(
    raw_residual: np.ndarray,
    scene: SceneBuffers,
    y_src: np.ndarray,
    y_pred: np.ndarray,
    settings: EngineSettings,
    eps: float = 1e-4,
) -> np.ndarray:
    """Extract shading-gated, locally centered multi-scale log-luminance detail for optional F3 mode."""
    sf = float(np.clip(max(scene.shape) / 1024.0, 0.5, 1.35))
    b_lo = _filter_log_field(raw_residual, 6.0 * sf, settings, eps_guide=0.25)
    b_hi = _filter_log_field(raw_residual, 22.0 * sf, settings, eps_guide=0.25)
    hf_clean = raw_residual - b_lo
    mid_raw = b_lo - b_hi
    local_dc = _filter_log_field(mid_raw, 18.0 * sf, settings, eps_guide=0.08)
    mid_centered = mid_raw - local_dc

    if scene.source_shading is not None:
        sh_gray = scene.source_shading if scene.source_shading.ndim == 2 else luminance(scene.source_shading)
        log_sh_blur = gaussian_filter(np.log(np.maximum(sh_gray, eps)).astype(np.float32), 2.5 * sf, mode='reflect')
        gx = sobel(log_sh_blur, axis=1, mode='reflect') / (8.0 * max(sf, 1.0))
        gy = sobel(log_sh_blur, axis=0, mode='reflect') / (8.0 * max(sf, 1.0))
        g_macro = gaussian_filter(np.hypot(gx, gy), 3.0 * sf, mode='reflect')
        w_shade = (1.0 / (1.0 + (g_macro / 0.055) ** 2)).astype(np.float32)
    else:
        w_shade = np.float32(1.0)

    w_conf = (1.0 / (1.0 + (np.abs(mid_raw) / 0.35) ** 2)).astype(np.float32)
    w_dark = np.clip((np.minimum(y_src, y_pred) - 0.008) / 0.04, 0.25, 1.0).astype(np.float32)
    hf_term = (0.85 * w_dark) * np.tanh(hf_clean / 0.85)
    mid_term = (0.75 * w_shade * w_conf) * np.tanh(mid_centered / 0.75)
    return (hf_term + mid_term).astype(np.float32)


def _get_or_compute_intrinsic_priors(
    scene: SceneBuffers,
    settings: EngineSettings,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (or fetch from content-keyed cache) light-independent intrinsic albedo_corr and detail_factor."""
    key = (
        _array_digest(scene.rgb_srgb),
        _array_digest(scene.albedo),
        _array_digest(scene.source_shading),
        _array_digest(scene.non_diffuse_residual),
        float(settings.base_blur_sigma),
        bool(settings.edge_aware_base),
        float(settings.detail_lock_strength),
        str(settings.detail_preservation_mode),
        scene.shape,
    )
    with _INTRINSIC_DETAIL_LOCK:
        cached = _INTRINSIC_DETAIL_CACHE.get(key)
        if cached is not None:
            _INTRINSIC_DETAIL_CACHE.move_to_end(key)
            return cached

    src = srgb_to_linear(scene.rgb_srgb)
    albedo_lin = srgb_to_linear(scene.albedo)
    albedo_corr = _preserve_source_chroma(src, albedo_lin, settings, eps=eps)

    y_src = np.maximum(luminance(src), eps)
    log_y_src = np.log(y_src)
    res_pos = (
        np.maximum(scene.non_diffuse_residual, 0.0)
        if scene.non_diffuse_residual is not None
        else None
    )
    if scene.source_shading is not None:
        sh = scene.source_shading[..., None] if scene.source_shading.ndim == 2 else scene.source_shading
        sh_3ch = np.broadcast_to(np.maximum(sh, eps), (*scene.shape, 3))
        pred_src = albedo_lin * sh_3ch
        if res_pos is not None:
            pred_src = pred_src + res_pos
        y_pred = np.maximum(luminance(pred_src), eps)
        # Log high-pass detail extraction is shift-invariant to arbitrary global scale on source_shading.
        raw_residual = log_y_src - np.log(y_pred)
        if settings.detail_preservation_mode == 'f3':
            log_detail = _compute_f3_log_detail(raw_residual, scene, y_src, y_pred, settings, eps=eps)
        else:
            log_detail = raw_residual - _compute_log_base(raw_residual, settings)
    else:
        if settings.detail_preservation_mode == 'f3':
            log_detail = _compute_f3_log_detail(log_y_src, scene, y_src, y_src, settings, eps=eps)
        else:
            log_detail = log_y_src - _compute_log_base(log_y_src, settings)
    detail_factor = np.exp(np.clip(log_detail, -1.5, 1.5) * settings.detail_lock_strength)[..., None].astype(np.float32)

    albedo_corr.setflags(write=False)
    detail_factor.setflags(write=False)
    with _INTRINSIC_DETAIL_LOCK:
        _INTRINSIC_DETAIL_CACHE[key] = (albedo_corr, detail_factor)
        _INTRINSIC_DETAIL_CACHE.move_to_end(key)
        while len(_INTRINSIC_DETAIL_CACHE) > _INTRINSIC_CACHE_MAX:
            _INTRINSIC_DETAIL_CACHE.popitem(last=False)
    return albedo_corr, detail_factor


def _get_or_compute_intrinsic_residual(
    scene: SceneBuffers,
    settings: EngineSettings,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Multiscale classification of non_diffuse_residual into fine detail, mid sheen, and broad source glare."""
    if scene.non_diffuse_residual is None:
        return None
    key = (
        _array_digest(scene.rgb_srgb),
        _array_digest(scene.non_diffuse_residual),
        float(settings.base_blur_sigma),
        bool(settings.edge_aware_base),
        scene.shape,
    )
    with _INTRINSIC_DETAIL_LOCK:
        cached = _INTRINSIC_RESIDUAL_CACHE.get(key)
        if cached is not None:
            _INTRINSIC_RESIDUAL_CACHE.move_to_end(key)
            return cached

    res_pos = np.maximum(scene.non_diffuse_residual, 0.0).astype(np.float32)
    sigma_fine = 3.0
    sigma_broad = min(float(settings.base_blur_sigma), 14.0)
    if settings.edge_aware_base:
        src = srgb_to_linear(scene.rgb_srgb)
        log_y_src = np.log(np.maximum(luminance(src), eps)).astype(np.float32)
        b_fine = np.stack([
            _joint_guided_filter(res_pos[..., c], log_y_src, sigma_fine, eps_guide=0.008)
            for c in range(3)
        ], axis=-1)
        b_broad = np.stack([
            _joint_guided_filter(res_pos[..., c], log_y_src, sigma_broad, eps_guide=0.012)
            for c in range(3)
        ], axis=-1)
    else:
        b_fine = np.stack([gaussian_filter(res_pos[..., c], sigma_fine, mode='reflect') for c in range(3)], axis=-1)
        b_broad = np.stack([gaussian_filter(res_pos[..., c], sigma_broad, mode='reflect') for c in range(3)], axis=-1)

    res_lobe = np.clip(b_broad, 0.0, res_pos).astype(np.float32)
    res_fine = np.maximum(res_pos - np.clip(b_fine, 0.0, res_pos), 0.0).astype(np.float32)
    res_mid = np.maximum(res_pos - res_fine - res_lobe, 0.0).astype(np.float32)
    for arr in (res_fine, res_mid, res_lobe):
        arr.setflags(write=False)
    with _INTRINSIC_DETAIL_LOCK:
        _INTRINSIC_RESIDUAL_CACHE[key] = (res_fine, res_mid, res_lobe)
        _INTRINSIC_RESIDUAL_CACHE.move_to_end(key)
        while len(_INTRINSIC_RESIDUAL_CACHE) > _INTRINSIC_CACHE_MAX:
            _INTRINSIC_RESIDUAL_CACHE.popitem(last=False)
    return res_fine, res_mid, res_lobe


def _modulate_intrinsic_residual(
    scene: SceneBuffers,
    src_lin: np.ndarray,
    target_diffuse: np.ndarray,
    geometry_confidence: np.ndarray,
    settings: EngineSettings,
    residual_layers: tuple[np.ndarray, np.ndarray, np.ndarray],
    eps: float = 1e-4,
) -> np.ndarray:
    """Modulate multiscale residual layers using shading ratio and decomposition certainty.

    - Fine non-diffuse texture (`res_fine`) responds to the new target illumination.
    - Broad original source glare (`res_lobe`) stays untinted so old reflections are never
      painted with the new light's color, and fades cleanly where confident relighting applies.
    - Uncertain regions (low geometry confidence or clipped highlights) avoid aggressive removal.
    """
    res_fine, res_mid, res_lobe = residual_layers
    t_pos = np.maximum(target_diffuse, 0.0)
    t_y = luminance(t_pos)

    min_c = float(settings.min_geometry_confidence)
    conf_geom = np.clip((geometry_confidence - min_c) / 0.35, 0.0, 1.0).astype(np.float32)
    sat_ambig = np.clip((np.max(src_lin, axis=-1) - 0.88) / 0.12, 0.0, 1.0).astype(np.float32)
    decomp_conf = conf_geom * (1.0 - 0.50 * sat_ambig)

    if scene.source_shading is not None:
        sh = scene.source_shading if scene.source_shading.ndim == 2 else luminance(scene.source_shading)
        active = (scene.depth > 0) & (scene.lightable_mask > 0.05)
        sh_med = max(float(np.median(sh[active])) if np.any(active) else float(np.median(sh)), eps)
        s_rel = np.maximum(sh / sh_med, eps)
        lobe_gate = np.clip(t_y / np.maximum(0.65 * s_rel, 0.35), 0.0, 1.15) * np.clip(t_y / 0.45, 0.0, 1.0)
    else:
        lobe_gate = np.clip(t_y / 0.65, 0.0, 1.25)

    # Confident zones suppress broad source glare aggressively in shadows; uncertain zones soften removal.
    lobe_weight = (0.12 * lobe_gate) * decomp_conf + (0.20 * np.clip(t_y / 0.50, 0.25, 1.0)) * (1.0 - decomp_conf)
    mid_weight = 0.20 * np.clip(0.35 + 0.65 * t_y, 0.25, 1.30)

    # Fine surface appearance (metal clasps, hair catchlights, wave/fabric micro-texture) preserves
    # crisp structure under ambient fill while brightening naturally with target illumination.
    fine_weight = 0.20 + 0.32 * np.clip(t_y, 0.0, 1.30)
    t_peak = np.maximum(np.max(t_pos, axis=-1, keepdims=True), eps)
    fine_tint = 0.45 + 0.55 * (t_pos / t_peak)
    fine_contrib = res_fine * (fine_weight[..., None] * fine_tint)
    mid_contrib = res_mid * mid_weight[..., None]
    lobe_contrib = res_lobe * lobe_weight[..., None]
    return (fine_contrib + mid_contrib + lobe_contrib).astype(np.float32)


def _compose_intrinsic_output(
    scene: SceneBuffers,
    albedo_corr: np.ndarray,
    detail_factor: np.ndarray,
    residual_layers: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    target_diffuse: np.ndarray,
    target_specular: np.ndarray,
    geometry_confidence: np.ndarray,
    settings: EngineSettings,
    eps: float = 1e-4,
    source_diffuse: np.ndarray | None = None,
    source_specular: np.ndarray | None = None,
    target_rig: LightingRig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Source-anchored intrinsic compositing core used by both compose_intrinsic() and RelightSession.

    Computes source-to-target lighting deltas via diffuse bounded gain and confidence-gated
    signed specular transfer anchored to the original photograph rather than replacing pixels
    with raw albedo * target_shading reconstructions.
    """
    if (
        source_diffuse is not None
        and not settings.preserve_source_lighting
        and abs(float(settings.output_exposure) - 1.0) < 1e-6
        and float(settings.atmosphere_density) <= 0.0
        and np.allclose(source_diffuse, target_diffuse, atol=1e-6)
        and (source_specular is None or np.allclose(source_specular, target_specular, atol=1e-6))
    ):
        return np.clip(scene.rgb_srgb, 0.0, 1.0).astype(np.float32), np.ones(scene.shape, dtype=np.float32)

    src = srgb_to_linear(scene.rgb_srgb)
    s_d_in = np.maximum(
        source_diffuse if source_diffuse is not None else np.ones_like(target_diffuse),
        eps,
    )
    t_pos, t_spec_eff = _attenuate_background_lighting(
        scene,
        target_diffuse,
        target_specular,
        settings,
        source_diffuse=s_d_in,
        albedo_lin=albedo_corr,
        src_lin=src,
        eps=eps,
    )
    t_pos = np.maximum(t_pos, 0.0)

    # 1. Controlled diffuse transfer anchored to the original photograph with unified perceptual chroma & anti-blowout soft-knee.
    s_y = np.maximum(luminance(s_d_in), eps)
    t_y = np.maximum(luminance(t_pos), eps)
    ratio = np.clip(t_y / s_y, settings.ratio_min, settings.ratio_max)

    min_gain = settings.ratio_min * settings.color_ratio_min
    max_gain = settings.ratio_max * settings.color_ratio_max
    t_chan_max = np.maximum(np.max(t_pos, axis=-1, keepdims=True), eps)
    t_hue = t_pos / t_chan_max
    t_sat = 1.0 - np.min(t_hue, axis=-1, keepdims=True)
    sat_w = np.clip((t_sat - 0.25) / 0.55, 0.0, 1.0)
    # Allow deep cast shadows (low target-to-source energy) to approach min_gain cleanly without washing out.
    deep_shadow_w = np.clip((0.15 - t_y / np.maximum(s_y, eps)) / 0.13, 0.0, 1.0)[..., None]
    eff_min_w = np.maximum(sat_w, deep_shadow_w)
    dyn_min_gain = float(settings.ratio_min) * (1.0 - eff_min_w) + min_gain * eff_min_w
    rgb_gain = np.clip(t_pos / s_d_in, dyn_min_gain, max_gain)
    if settings.lighting_smooth_sigma > 0.0:
        ratio = np.clip(_depth_guided_smooth(ratio, scene.depth, settings.lighting_smooth_sigma), settings.ratio_min, settings.ratio_max)
        rgb_gain = np.clip(_depth_guided_smooth(rgb_gain, scene.depth, settings.lighting_smooth_sigma), min_gain, max_gain)

    y_src = np.maximum(luminance(src), eps)
    min_c = float(settings.min_geometry_confidence)
    geom_conf_w = np.clip((geometry_confidence - min_c) / 0.30, 0.0, 1.0).astype(np.float32)

    ratio, rgb_gain = _soft_knee_bright_ratio(ratio, rgb_gain, y_src, eps=eps)

    change_w = np.ones(scene.shape, dtype=np.float32)
    ratio, rgb_gain = _apply_perceptual_color_anchor(
        src,
        y_src,
        s_d_in,
        t_pos,
        ratio,
        rgb_gain,
        geom_conf_w,
        change_w,
        settings,
        eps=eps,
    )

    # Near-black stability guard: prevent unstable noise amplification when original luminance is near zero.
    near_black_damp = np.clip((y_src - 0.002) / 0.015, 0.25, 1.0)[..., None]
    rgb_gain_safe = np.where(rgb_gain > 1.0, 1.0 + (rgb_gain - 1.0) * near_black_damp, rgb_gain)

    # Separate broad source specular glare cleanly so residual handling never subtracts crisp surface details.
    src_spec_in = np.maximum(source_specular, 0.0) if source_specular is not None else np.zeros_like(target_specular)
    lossless_detail = bool(getattr(settings, "lossless_ratio_detail", False))
    if residual_layers is not None:
        _, _, res_lobe = residual_layers
        if lossless_detail:
            # Keep 100% of crisp high-frequency textures/dial/brushed metal in src_diff_anchor;
            # only separate smooth macro glare lobes (sigma=6) weighted away from fine detail edges.
            smooth_lobe = gaussian_filter(res_lobe, (6.0, 6.0, 0.0), mode="reflect")
            g_src_hf = np.abs(y_src - gaussian_filter(y_src, 1.5, mode="reflect"))
            crisp_keep = np.clip(g_src_hf / 0.04, 0.0, 1.0)[..., None]
            eff_lobe = smooth_lobe * (1.0 - 0.85 * crisp_keep)
        else:
            eff_lobe = res_lobe
        if scene.source_shading is not None:
            sh = scene.source_shading[..., None] if scene.source_shading.ndim == 2 else scene.source_shading
            pred_diff = albedo_corr * np.broadcast_to(np.maximum(sh, eps), (*scene.shape, 3))
        else:
            pred_diff = albedo_corr * 0.5
        max_glare = 0.28 if lossless_detail else 0.55
        glare_frac = np.clip(eff_lobe / np.maximum(pred_diff + eff_lobe, eps), 0.0, max_glare)
        removed_glare = glare_frac * src
        src_diff_anchor = np.maximum(src - removed_glare, eps)
        # Avoid double-subtracting glare already accounted for by residual separation.
        spec_src_eff = np.maximum(src_spec_in - removed_glare, 0.0)
        lobe_scale = np.clip(t_y / np.maximum(s_y, eps), 0.0, 1.25) * np.clip(t_y / (0.24 if lossless_detail else 0.40), 0.0, 1.0)
        sat_ambig = np.clip((np.max(src, axis=-1) - 0.88) / 0.12, 0.0, 1.0).astype(np.float32)
        decomp_conf = geom_conf_w * (1.0 - 0.50 * sat_ambig)
        lobe_keep = lobe_scale * decomp_conf + np.clip(t_y / 0.50, 0.25, 1.0) * (1.0 - decomp_conf)
        if lossless_detail:
            lobe_keep = np.maximum(lobe_keep, 0.72)
        t_peak = np.maximum(np.max(t_pos, axis=-1, keepdims=True), eps)
        glare_tint = 0.25 + 0.75 * (t_pos / t_peak)
        relit_base = src_diff_anchor * rgb_gain_safe + removed_glare * (lobe_keep[..., None] * glare_tint)
    else:
        spec_src_eff = src_spec_in
        relit_base = src * rgb_gain_safe

    # 2. Controlled signed specular delta (target - source), restricted by geometry & material confidence.
    mat_conf_w = (
        np.clip(0.35 + 0.65 * np.asarray(scene.material_confidence, dtype=np.float32), 0.0, 1.0)
        if scene.material_confidence is not None
        else np.ones(scene.shape, dtype=np.float32)
    )
    spec_weight = (geom_conf_w * mat_conf_w)[..., None] * settings.specular_scale
    signed_spec = (np.maximum(t_spec_eff, 0.0) - spec_src_eff) * spec_weight
    reloc_s = float(np.clip(settings.highlight_relocation_strength, 0.0, 2.0))
    if reloc_s > 0.0:
        pos_mask = signed_spec > 0.0
        signed_spec = np.where(pos_mask, signed_spec * (1.0 + 0.55 * reloc_s), signed_spec)
    max_pos = np.maximum(relit_base * (1.5 + 1.0 * reloc_s) + 0.45 * np.maximum(settings.specular_scale, 0.0), eps)
    pos_spec = max_pos * np.tanh(np.maximum(signed_spec, 0.0) / max_pos)
    neg_budget = -np.minimum(
        0.75 * relit_base,
        np.maximum(src_spec_in, 0.25 * relit_base) * np.clip(geom_conf_w * mat_conf_w, 0.0, 1.0)[..., None],
    )
    neg_spec = np.maximum(np.minimum(signed_spec, 0.0), neg_budget)
    relit = np.maximum(relit_base + pos_spec + neg_spec, 0.0) * settings.output_exposure

    if settings.detail_preservation_mode == "wavelet":
        detail_w = _compute_detail_lock_weight_map(
            scene, geometry_confidence, ratio, residual_layers, settings, eps=eps
        )
        relit = compose_wavelet_identity_lock(
            src,
            relit,
            levels=settings.wavelet_levels,
            detail_lock_strength=settings.detail_lock_strength,
            mid_blend=settings.wavelet_mid_blend,
            detail_weight_map=detail_w,
            eps=eps,
        )

    mask = _active_blend_mask(scene, geometry_confidence, settings)
    out = _finalize_protected_display_output(scene, relit, src, mask, settings, eps=eps)
    return out, ratio.astype(np.float32)


def compose_intrinsic(
    scene: SceneBuffers,
    target_diffuse: np.ndarray,
    target_specular: np.ndarray,
    geometry_confidence: np.ndarray,
    settings: EngineSettings,
    *,
    source_diffuse: np.ndarray | None = None,
    source_specular: np.ndarray | None = None,
    target_rig: LightingRig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-render from estimated albedo + target shading while locking source high-frequency detail."""
    if scene.albedo is None:
        raise ValueError("compose_intrinsic requires scene.albedo.")
    eps = 1e-4
    albedo_corr, detail_factor = _get_or_compute_intrinsic_priors(scene, settings, eps=eps)
    residual_layers = _get_or_compute_intrinsic_residual(scene, settings, eps=eps)
    return _compose_intrinsic_output(
        scene,
        albedo_corr,
        detail_factor,
        residual_layers,
        target_diffuse,
        target_specular,
        geometry_confidence,
        settings,
        eps=eps,
        source_diffuse=source_diffuse,
        source_specular=source_specular,
        target_rig=target_rig,
    )


def compose_hybrid_neural(
    scene: SceneBuffers,
    neural_rgb_srgb: np.ndarray,
    geometry_confidence: np.ndarray,
    settings: EngineSettings,
) -> np.ndarray:
    """Lock high-frequency photographic texture, protected masks, and alpha over a neural relight."""
    src = srgb_to_linear(scene.rgb_srgb)
    neural_lin = srgb_to_linear(neural_rgb_srgb)
    eps = 1e-4

    if settings.detail_preservation_mode == "wavelet":
        relit = compose_wavelet_identity_lock(
            src,
            neural_lin,
            levels=settings.wavelet_levels,
            detail_lock_strength=settings.detail_lock_strength,
            mid_blend=settings.wavelet_mid_blend,
            eps=eps,
        ) * settings.output_exposure
    else:
        y_src = np.maximum(luminance(src), eps)
        log_y_src = np.log(y_src)
        log_detail_src = np.clip(log_y_src - _compute_log_base(log_y_src, settings), -1.5, 1.5)

        y_neu = np.maximum(luminance(neural_lin), eps)
        log_y_neu = np.log(y_neu)
        log_base_neu = _compute_log_base(log_y_neu, settings)
        log_detail_neu = np.clip(log_y_neu - log_base_neu, -1.5, 1.5)

        alpha_detail = float(settings.detail_lock_strength)
        log_y_locked = log_base_neu + alpha_detail * log_detail_src + (1.0 - alpha_detail) * log_detail_neu
        y_locked = np.maximum(np.exp(log_y_locked), eps)
        relit = neural_lin * (y_locked / y_neu)[..., None] * settings.output_exposure

    mask = _active_blend_mask(scene, geometry_confidence, settings)
    return _finalize_protected_display_output(scene, relit, src, mask, settings, eps=eps)
