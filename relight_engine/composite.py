from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from .color import srgb_to_linear, linear_to_srgb, luminance
from .contracts import SceneBuffers, EngineSettings

def compose_source_preserving(scene: SceneBuffers, source_diffuse: np.ndarray,
                              target_diffuse: np.ndarray, source_specular: np.ndarray,
                              target_specular: np.ndarray, geometry_confidence: np.ndarray,
                              shadow_visibility: np.ndarray, settings: EngineSettings) -> tuple[np.ndarray, np.ndarray]:
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
    color_ratio = np.power(color_ratio, 0.65)

    y = np.maximum(luminance(src), eps)
    log_y = np.log(y)
    base = np.exp(gaussian_filter(log_y, settings.base_blur_sigma, mode='reflect'))
    detail = y - base
    y_relit = base * ratio + detail * np.power(ratio, settings.detail_power)
    y_relit = np.maximum(y_relit, eps)
    relit = src * (y_relit / y)[..., None] * color_ratio
    relit *= settings.output_exposure

    # Add only the incremental GGX highlight predicted by the target rig.
    spec_delta = np.maximum(target_specular - source_specular, 0.0)
    relit += spec_delta * settings.specular_scale

    # Confidence- and semantic-aware blend: uncertain geometry returns toward the original image.
    mask = np.clip(scene.lightable_mask, 0, 1)
    if scene.alpha_mask is not None:
        mask *= np.clip(scene.alpha_mask, 0, 1)
    mask *= np.clip(geometry_confidence, 0, 1)
    mask *= (geometry_confidence >= settings.min_geometry_confidence).astype(np.float32)
    mask = np.clip(mask, 0, 1)
    relit = src * (1 - mask[..., None]) + relit * mask[..., None]

    # Soft shoulder; avoid hard clipping without globally flattening the image.
    peak = np.max(relit, axis=-1)
    source_peak = np.max(src, axis=-1)
    threshold, width = 0.94, 0.08
    compressed = threshold + width * (1 - np.exp(-np.maximum(peak - threshold, 0) / width))
    # Do not alter an unchanged source image: shoulder only newly added highlight energy.
    newly_hot = (peak > threshold) & (peak > source_peak + 1e-6)
    scale = np.where(newly_hot, compressed / np.maximum(peak, eps), 1.0)
    relit *= scale[..., None]
    out = linear_to_srgb(relit)
    return out.astype(np.float32), ratio.astype(np.float32)
