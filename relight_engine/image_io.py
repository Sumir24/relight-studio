"""Robust image decoding and color normalization for heterogeneous photo inputs."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import io
import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps


ALPHA_CONVENTION = "straight"


@dataclass
class ImageInput:
    rgb_srgb: np.ndarray
    alpha_mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        rgb = np.asarray(self.rgb_srgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("rgb_srgb must have shape HxWx3.")
        if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 1):
            raise ValueError("rgb_srgb must be finite and in [0,1].")
        if self.alpha_mask is not None:
            alpha = np.asarray(self.alpha_mask)
            if alpha.shape != rgb.shape[:2] or not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
                raise ValueError("alpha_mask must be finite HxW in [0,1].")


def _apply_exif_orientation_array(arr: np.ndarray, orientation: int | None) -> np.ndarray:
    """Apply EXIF orientation transform to an HxWxC float32 array."""
    if orientation == 2:
        return np.ascontiguousarray(arr[:, ::-1])
    if orientation == 3:
        return np.ascontiguousarray(arr[::-1, ::-1])
    if orientation == 4:
        return np.ascontiguousarray(arr[::-1, :])
    if orientation == 5:
        return np.ascontiguousarray(np.swapaxes(arr, 0, 1))
    if orientation == 6:
        return np.ascontiguousarray(np.rot90(arr, k=3))
    if orientation == 7:
        return np.ascontiguousarray(np.rot90(arr[:, ::-1], k=1))
    if orientation == 8:
        return np.ascontiguousarray(np.rot90(arr, k=1))
    return arr


def load_image(path: str | Path) -> ImageInput:
    """Load an 8-bit or 16-bit raster image, apply EXIF orientation and convert ICC to sRGB.

    Alpha follows a single explicit convention (`ALPHA_CONVENTION = "straight"`): transparency
    is preserved as a separate straight [0, 1] alpha mask without background premultiplication.
    """
    source = Path(path)
    with Image.open(source) as opened:
        original_mode = opened.mode
        original_size = opened.size
        source_format = opened.format
        exif = opened.getexif() if hasattr(opened, "getexif") else None
        orientation = int(exif.get(274, 1)) if exif else 1
        icc_data = opened.info.get("icc_profile")

        # Try 16-bit lossless decode via OpenCV when available (Pillow truncates RGB 16-bit PNGs to 8-bit).
        raw_cv = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if raw_cv is not None and raw_cv.dtype == np.uint16:
            arr_u16 = _apply_exif_orientation_array(raw_cv, orientation)
            if arr_u16.ndim == 2:
                rgb_f = np.repeat(arr_u16[..., None].astype(np.float32) / 65535.0, 3, axis=-1)
                alpha = None
            elif arr_u16.shape[-1] == 4:
                bgra_f = arr_u16.astype(np.float32) / 65535.0
                rgb_f = bgra_f[..., [2, 1, 0]]
                alpha = bgra_f[..., 3]
            else:
                rgb_f = arr_u16[..., :3][..., ::-1].astype(np.float32) / 65535.0
                alpha = None
            bit_depth = 16
            icc_applied = False
            if icc_data:
                try:
                    source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_data))
                    srgb_profile = ImageCms.createProfile("sRGB")
                    preview_u8 = Image.fromarray(np.round(np.clip(rgb_f, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
                    conv_u8 = ImageCms.profileToProfile(preview_u8, source_profile, srgb_profile, outputMode="RGB")
                    rgb_f = np.asarray(conv_u8, dtype=np.float32) / 255.0
                    icc_applied = True
                except Exception as exc:
                    raise ValueError(f"Could not convert embedded ICC profile to sRGB for {source.name}: {exc}") from exc
            rgb = rgb_f
        else:
            image = ImageOps.exif_transpose(opened)
            icc_data = image.info.get("icc_profile") or icc_data
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            rgba = image.convert("RGBA")
            alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0 if has_alpha else None
            rgb_image = rgba.convert("RGB")
            icc_applied = False
            if icc_data:
                try:
                    source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_data))
                    srgb_profile = ImageCms.createProfile("sRGB")
                    rgb_image = ImageCms.profileToProfile(
                        rgb_image, source_profile, srgb_profile, outputMode="RGB"
                    )
                    icc_applied = True
                except Exception as exc:
                    raise ValueError(f"Could not convert embedded ICC profile to sRGB for {source.name}: {exc}") from exc
            rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
            bit_depth = 8

    result = ImageInput(
        rgb_srgb=rgb.astype(np.float32),
        alpha_mask=None if alpha is None or np.all(alpha >= 1.0) else alpha.astype(np.float32),
        metadata={
            "source_path": str(source.resolve()),
            "format": source_format,
            "original_mode": original_mode,
            "bit_depth": bit_depth,
            "alpha_convention": ALPHA_CONVENTION,
            "original_size": list(original_size),
            "oriented_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "exif_orientation_applied": True,
            "icc_profile_present": bool(icc_data),
            "icc_to_srgb_applied": icc_applied,
            "alpha_present": alpha is not None,
            "alpha_preserved_separately": alpha is not None,
        },
    )
    result.validate()
    return result


def save_image(
    path: str | Path,
    rgb_srgb: np.ndarray,
    *,
    alpha_mask: np.ndarray | None = None,
    bit_depth: int = 8,
) -> Path:
    """Save sRGB [0,1] image to disk in 8-bit or 16-bit with straight alpha transparency."""
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb_c = np.clip(np.asarray(rgb_srgb, dtype=np.float32), 0.0, 1.0)
    if bit_depth == 16:
        rgb_u16 = np.round(rgb_c * 65535.0).astype(np.uint16)
        if alpha_mask is not None:
            alpha_u16 = np.round(np.clip(np.asarray(alpha_mask, dtype=np.float32), 0.0, 1.0) * 65535.0).astype(np.uint16)
            bgra = np.dstack([rgb_u16[..., 2], rgb_u16[..., 1], rgb_u16[..., 0], alpha_u16])
            ok = cv2.imwrite(str(out), bgra)
        else:
            bgr = np.dstack([rgb_u16[..., 2], rgb_u16[..., 1], rgb_u16[..., 0]])
            ok = cv2.imwrite(str(out), bgr)
        if not ok:
            raise ValueError(f"Failed to encode 16-bit image at {out}")
        return out

    rgb_u8 = np.round(rgb_c * 255.0).astype(np.uint8)
    if alpha_mask is not None:
        alpha_u8 = np.round(np.clip(alpha_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(np.dstack([rgb_u8, alpha_u8]), mode="RGBA").save(out)
    else:
        Image.fromarray(rgb_u8, mode="RGB").save(out)
    return out
