from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from .color import srgb_to_linear, linear_to_srgb, luminance
from .contracts import SceneBuffers, EngineSettings

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


def _active_blend_mask(
    scene: SceneBuffers, geometry_confidence: np.ndarray, settings: EngineSettings
) -> np.ndarray:
    mask = np.clip(scene.lightable_mask, 0, 1).astype(np.float32, copy=True)
    if scene.protected_mask is not None:
        mask *= 1.0 - np.clip(scene.protected_mask, 0, 1)
    if scene.alpha_mask is not None:
        mask *= np.clip(scene.alpha_mask, 0, 1)
    mask *= np.clip(geometry_confidence, 0, 1)
    mask *= (geometry_confidence >= settings.min_geometry_confidence).astype(np.float32)
    return np.clip(mask, 0, 1).astype(np.float32)


def _apply_highlight_shoulder(relit: np.ndarray, src: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    peak = np.max(relit, axis=-1)
    source_peak = np.max(src, axis=-1)
    threshold, width = 0.94, 0.08
    compressed = threshold + width * (1 - np.exp(-np.maximum(peak - threshold, 0) / width))
    newly_hot = (peak > threshold) & (peak > source_peak + 1e-6)
    scale = np.where(newly_hot, compressed / np.maximum(peak, eps), 1.0)
    return relit * scale[..., None]


def compose_source_preserving(scene: SceneBuffers, source_diffuse: np.ndarray,
                              target_diffuse: np.ndarray, source_specular: np.ndarray,
                              target_specular: np.ndarray, geometry_confidence: np.ndarray,
                              settings: EngineSettings) -> tuple[np.ndarray, np.ndarray]:
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

    y = np.maximum(luminance(src), eps)
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
