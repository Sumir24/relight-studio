"""Stationary (à-trous) B3-spline wavelet decomposition and identity-lock compositing."""
from __future__ import annotations

import cv2
import numpy as np

from .color import luminance

_B3_KERNEL_1D = np.array([1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0], dtype=np.float32)


def _dilated_b3_kernel(level: int) -> np.ndarray:
    """Return 1D B3-spline kernel with 2**level - 1 zeros inserted between taps."""
    step = 1 << level
    k_len = 4 * step + 1
    kernel = np.zeros((k_len,), dtype=np.float32)
    kernel[0::step] = _B3_KERNEL_1D
    return kernel


def _smooth_atrous(arr: np.ndarray, level: int) -> np.ndarray:
    """Apply separable à-trous B3 smoothing at dyadic scale 2**level."""
    kernel = _dilated_b3_kernel(level)
    if arr.ndim == 2:
        return cv2.sepFilter2D(arr, cv2.CV_32F, kernel, kernel, borderType=cv2.BORDER_REFLECT_101)
    channels = [
        cv2.sepFilter2D(arr[..., c], cv2.CV_32F, kernel, kernel, borderType=cv2.BORDER_REFLECT_101)
        for c in range(arr.shape[-1])
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def atrous_decompose(image: np.ndarray, levels: int = 4) -> tuple[list[np.ndarray], np.ndarray]:
    """Decompose a 2D or HxWxC float32 image into exact-reconstruction B3 wavelet bands.

    Returns (detail_bands, residual_low) such that sum(detail_bands) + residual_low == image.
    """
    if levels < 1:
        raise ValueError("levels must be >= 1.")
    current = np.asarray(image, dtype=np.float32)
    details: list[np.ndarray] = []
    for lvl in range(levels):
        nxt = _smooth_atrous(current, lvl)
        details.append((current - nxt).astype(np.float32))
        current = nxt
    return details, current.astype(np.float32)


def atrous_reconstruct(detail_bands: list[np.ndarray], residual_low: np.ndarray) -> np.ndarray:
    """Reconstruct an image from its à-trous wavelet bands and low-frequency residual."""
    out = np.array(residual_low, dtype=np.float32, copy=True)
    for band in detail_bands:
        out += band
    return out


def compose_wavelet_identity_lock(
    src_lin: np.ndarray,
    relit_lin: np.ndarray,
    *,
    levels: int = 4,
    detail_lock_strength: float = 0.95,
    mid_blend: float = 0.55,
    chroma_lock_strength: float = 0.75,
    detail_weight_map: np.ndarray | None = None,
    eps: float = 1e-4,
) -> np.ndarray:
    """Lock original multi-scale texture and fine chroma onto a relit image in wavelet space.

    Uses log-luminance à-trous bands so multiplicative surface texture scales proportionally
    with the relit illumination without shadow over-sharpening or boundary halos, while
    preserving broad lighting and color temperature from `relit_lin`. When `detail_weight_map`
    is provided (HxW in [0, 1]), copying is reduced around confident illumination boundaries
    and source lighting artifacts so baked shadows/glare are not copied back.
    """
    src_pos = np.maximum(np.asarray(src_lin, dtype=np.float32), eps)
    relit_pos = np.maximum(np.asarray(relit_lin, dtype=np.float32), eps)

    y_src = np.maximum(luminance(src_pos), eps)
    y_relit = np.maximum(luminance(relit_pos), eps)
    log_y_src = np.log(y_src).astype(np.float32)
    log_y_relit = np.log(y_relit).astype(np.float32)

    src_bands, _ = atrous_decompose(log_y_src, levels=levels)
    rel_bands, rel_low = atrous_decompose(log_y_relit, levels=levels)

    alpha = float(np.clip(detail_lock_strength, 0.0, 1.0))
    mid_w = float(np.clip(mid_blend, 0.0, 1.0))
    # Reduce luminance wavelet lock in near-black pixels so sensor/JPEG noise isn't amplified in log space.
    lum_gate = np.clip((np.minimum(y_src, y_relit) - 0.005) / 0.03, 0.30, 1.0).astype(np.float32)
    d_weight = (
        np.clip(np.asarray(detail_weight_map, dtype=np.float32), 0.0, 1.0)
        if detail_weight_map is not None
        else None
    )

    # Preserve fine micro-contrast when metallic/shaded relighting lowers base luminance below source.
    fine_gain = np.where(
        y_relit < y_src,
        1.0 + 0.80 * np.clip(np.sqrt(y_src / np.maximum(y_relit, eps)) - 1.0, 0.0, 0.90) * np.clip((y_relit - 0.020) / 0.09, 0.0, 1.0),
        1.0,
    ).astype(np.float32)

    blended_bands: list[np.ndarray] = []
    for idx in range(levels):
        if idx == 0:
            w_src = alpha
        elif idx == 1:
            w_src = alpha * 0.85
        else:
            frac = float(idx - 1) / float(max(levels - 1, 1))
            w_src = alpha * (1.0 - frac) * (1.0 - mid_w) + (1.0 - mid_w) * 0.35 * frac
            w_src = float(np.clip(w_src, 0.0, alpha))
        w_eff = w_src * lum_gate
        if d_weight is not None:
            # Keep crisp micro-grain/typography locked on fine bands while releasing macro artifact bands.
            band_gate = (0.55 + 0.45 * d_weight) if idx == 0 else d_weight
            w_eff = w_eff * band_gate
        src_band = src_bands[idx]
        if idx < 2:
            # Avoid dulling aligned specular/high-contrast relit details that already exceed source log contrast.
            aligned_stronger = (rel_bands[idx] * src_band > 0.0) & (np.abs(rel_bands[idx]) > np.abs(src_band))
            src_target = np.where(
                aligned_stronger,
                0.55 * src_band * fine_gain + 0.45 * rel_bands[idx],
                src_band * fine_gain,
            )
            band = w_eff * src_target + (1.0 - w_eff) * rel_bands[idx]
        else:
            band = w_eff * src_band + (1.0 - w_eff) * rel_bands[idx]
        # Clamp extreme log-detail amplification so harsh shadows stay clean.
        limit = 1.6 if idx < 2 else 2.0
        blended_bands.append(np.clip(band, -limit, limit).astype(np.float32))

    log_y_locked = atrous_reconstruct(blended_bands, rel_low)
    y_locked = np.maximum(np.exp(np.clip(log_y_locked, -12.0, 6.0)), eps).astype(np.float32)

    # Lock high-frequency chromaticity (levels 0..1) from source while taking broad light hue from relit.
    chroma_strength = float(np.clip(chroma_lock_strength, 0.0, 1.0)) * alpha
    if chroma_strength > 0.0:
        log_c_src = np.log(src_pos) - log_y_src[..., None]
        log_c_rel = np.log(relit_pos) - log_y_relit[..., None]
        c_src_bands, _ = atrous_decompose(log_c_src.astype(np.float32), levels=min(2, levels))
        c_rel_bands, c_rel_low = atrous_decompose(log_c_rel.astype(np.float32), levels=min(2, levels))
        chroma_conf = np.clip((np.minimum(y_src, y_relit) - 0.01) / 0.06, 0.0, 1.0)[..., None]
        if d_weight is not None:
            chroma_conf = chroma_conf * (0.45 + 0.55 * d_weight[..., None])
        c_locked_bands = []
        for idx in range(len(c_src_bands)):
            w_c = chroma_strength * (1.0 if idx == 0 else 0.70) * chroma_conf
            c_locked_bands.append((w_c * c_src_bands[idx] + (1.0 - w_c) * c_rel_bands[idx]).astype(np.float32))
        log_c_locked = atrous_reconstruct(c_locked_bands, c_rel_low)
        # Convert normalized log-chroma + locked luminance back to linear RGB, preserving y_locked.
        rgb_dir = np.exp(np.clip(log_c_locked, -3.0, 3.0)).astype(np.float32)
        dir_y = np.maximum(luminance(rgb_dir), eps)[..., None]
        out = rgb_dir * (y_locked[..., None] / dir_y)
    else:
        out = relit_pos * (y_locked / y_relit)[..., None]

    return np.maximum(out, 0.0).astype(np.float32)
