"""Robust image decoding and color normalization for heterogeneous photo inputs."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import io
import numpy as np
from PIL import Image, ImageCms, ImageOps


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


def load_image(path: str | Path) -> ImageInput:
    """Load a common raster image, apply EXIF orientation and convert ICC to sRGB.

    Transparency is preserved as a separate alpha mask; it is not flattened onto an
    arbitrary background. The lighting compositor later keeps fully transparent pixels
    unchanged and the output workflow restores the original alpha channel.
    """
    source = Path(path)
    with Image.open(source) as opened:
        original_mode = opened.mode
        original_size = opened.size
        source_format = opened.format
        image = ImageOps.exif_transpose(opened)
        icc_data = image.info.get("icc_profile")
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
    result = ImageInput(
        rgb_srgb=rgb.astype(np.float32),
        alpha_mask=None if alpha is None or np.all(alpha >= 1.0) else alpha.astype(np.float32),
        metadata={
            "source_path": str(source.resolve()),
            "format": source_format,
            "original_mode": original_mode,
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
