from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from .color import srgb_to_linear, linear_to_srgb, luminance
from .contracts import SceneBuffers, EngineSettings, LightingRig

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


def _depth_guided_smooth(
    field: np.ndarray, depth: np.ndarray, sigma: float, eps_guide: float = 0.008
) -> np.ndarray:
    """Smooth a scalar or HxWxC lighting field while preserving sharp depth discontinuities."""
    if sigma <= 0.0:
        return field
    guide = np.log(np.maximum(depth, 1e-3)).astype(np.float32)
    mean_i = gaussian_filter(guide, sigma, mode='reflect')
    mean_ii = gaussian_filter(guide * guide, sigma, mode='reflect')
    var_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)

    def _filter_2d(p: np.ndarray) -> np.ndarray:
        mean_p = gaussian_filter(p, sigma, mode='reflect')
        mean_ip = gaussian_filter(guide * p, sigma, mode='reflect')
        cov_ip = mean_ip - mean_i * mean_p
        a = cov_ip / (var_i + eps_guide)
        b = mean_p - a * mean_i
        return gaussian_filter(a, sigma, mode='reflect') * guide + gaussian_filter(b, sigma, mode='reflect')

    if field.ndim == 2:
        return _filter_2d(field.astype(np.float32)).astype(np.float32)
    channels = [_filter_2d(field[..., c].astype(np.float32)) for c in range(field.shape[-1])]
    return np.stack(channels, axis=-1).astype(np.float32)


def _active_blend_mask(
    scene: SceneBuffers, geometry_confidence: np.ndarray, settings: EngineSettings
) -> np.ndarray:
    mask = np.clip(scene.lightable_mask, 0, 1).astype(np.float32, copy=True)
    if scene.protected_mask is not None:
        mask *= 1.0 - np.clip(scene.protected_mask, 0, 1)
    if scene.alpha_mask is not None:
        mask *= np.clip(scene.alpha_mask, 0, 1)
    conf = np.clip(geometry_confidence, 0, 1).astype(np.float32)
    min_c = float(settings.min_geometry_confidence)
    ramp_hi = max(min_c + 0.15, 0.30)
    if ramp_hi > min_c + 1e-6:
        conf_weight = np.clip((conf - min_c) / (ramp_hi - min_c), 0.0, 1.0)
    else:
        conf_weight = (conf >= min_c).astype(np.float32)
    mask *= conf_weight * (conf >= min_c).astype(np.float32)
    return np.clip(mask, 0, 1).astype(np.float32)


def _apply_highlight_shoulder(relit: np.ndarray, src: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    peak = np.max(relit, axis=-1)
    source_peak = np.max(src, axis=-1)
    threshold, width = 0.94, 0.08
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


def compose_source_preserving(scene: SceneBuffers, source_diffuse: np.ndarray,
                              target_diffuse: np.ndarray, source_specular: np.ndarray,
                              target_specular: np.ndarray, geometry_confidence: np.ndarray,
                              settings: EngineSettings,
                              target_rig: LightingRig | None = None) -> tuple[np.ndarray, np.ndarray]:
    src = srgb_to_linear(scene.rgb_srgb)
    eps = 1e-4
    # Use a luminance ratio for stable detail preservation; carry a restrained RGB chroma shift separately.
    s_y = np.maximum(luminance(source_diffuse), eps)
    t_y = np.maximum(luminance(target_diffuse), eps)
    raw_ratio = t_y / s_y
    ratio = np.clip(raw_ratio, settings.ratio_min, settings.ratio_max)
    src_chroma = source_diffuse / s_y[..., None]
    tgt_chroma = target_diffuse / t_y[..., None]
    color_ratio = np.clip(tgt_chroma / np.maximum(src_chroma, eps),
                          settings.color_ratio_min, settings.color_ratio_max)
    color_ratio = np.power(color_ratio, 0.85)
    if settings.lighting_smooth_sigma > 0.0:
        ratio = np.clip(
            _depth_guided_smooth(ratio, scene.depth, settings.lighting_smooth_sigma),
            settings.ratio_min, settings.ratio_max,
        )
        color_ratio = np.clip(
            _depth_guided_smooth(color_ratio, scene.depth, settings.lighting_smooth_sigma),
            settings.color_ratio_min, settings.color_ratio_max,
        )

    y = np.maximum(luminance(src), eps)
    if abs(float(settings.detail_power) - 1.0) < 1e-6:
        y_relit = np.maximum(y * ratio, eps)
    else:
        log_y = np.log(y)
        base = np.exp(_compute_log_base(log_y, settings))
        detail = y - base
        y_relit = base * ratio + detail * np.power(ratio, settings.detail_power)
        y_relit = np.maximum(y_relit, eps)
    relit = src * (y_relit / y)[..., None] * color_ratio
    relit *= settings.output_exposure

    # Add only the incremental GGX highlight predicted by the target rig.
    spec_delta = np.maximum(target_specular - source_specular, 0.0)
    relit += spec_delta * settings.specular_scale
    relit = _apply_aerial_perspective(relit, scene, target_rig, settings)

    mask = _active_blend_mask(scene, geometry_confidence, settings)
    relit = src * (1 - mask[..., None]) + relit * mask[..., None]
    relit = _apply_highlight_shoulder(relit, src, eps)
    out = linear_to_srgb(relit)
    return out.astype(np.float32), ratio.astype(np.float32)


def compose_intrinsic(
    scene: SceneBuffers,
    target_diffuse: np.ndarray,
    target_specular: np.ndarray,
    geometry_confidence: np.ndarray,
    settings: EngineSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-render from estimated albedo + target shading while locking source high-frequency detail."""
    if scene.albedo is None:
        raise ValueError("compose_intrinsic requires scene.albedo.")
    src = srgb_to_linear(scene.rgb_srgb)
    albedo_lin = srgb_to_linear(scene.albedo)
    eps = 1e-4

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
        # Log high-pass detail extraction is shift-invariant to arbitrary global scale on source_shading.
        raw_residual = log_y_src - np.log(np.maximum(luminance(pred_src), eps))
        log_detail = raw_residual - _compute_log_base(raw_residual, settings)
    else:
        log_detail = log_y_src - _compute_log_base(log_y_src, settings)
    detail_factor = np.exp(np.clip(log_detail, -1.5, 1.5) * settings.detail_lock_strength)[..., None]

    base_relit = albedo_lin * np.maximum(target_diffuse, 0.0)
    if res_pos is not None:
        base_relit = base_relit + res_pos * 0.25
    relit = base_relit * detail_factor * settings.output_exposure
    relit = relit + np.maximum(target_specular, 0.0) * settings.specular_scale

    mask = _active_blend_mask(scene, geometry_confidence, settings)
    relit = src * (1 - mask[..., None]) + relit * mask[..., None]
    relit = _apply_highlight_shoulder(relit, src, eps)

    t_y = np.maximum(luminance(target_diffuse), eps)
    if scene.source_shading is not None:
        sh = scene.source_shading[..., None] if scene.source_shading.ndim == 2 else scene.source_shading
        s_y = np.maximum(luminance(np.broadcast_to(sh, (*scene.shape, 3))), eps)
        ratio = np.clip(t_y / s_y, settings.ratio_min, settings.ratio_max)
    else:
        ratio = np.clip(t_y, settings.ratio_min, settings.ratio_max)
    return linear_to_srgb(relit).astype(np.float32), ratio.astype(np.float32)


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
    relit = src * (1 - mask[..., None]) + relit * mask[..., None]
    relit = _apply_highlight_shoulder(relit, src, eps)
    return linear_to_srgb(relit).astype(np.float32)
