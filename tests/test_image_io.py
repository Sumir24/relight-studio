from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from relight_engine.image_io import load_image


def test_image_loader_preserves_alpha_instead_of_flattening(tmp_path):
    rgba = np.zeros((8, 10, 4), dtype=np.uint8)
    rgba[..., 0] = 220
    rgba[..., 1] = 80
    rgba[..., 2] = 30
    rgba[:3, :, 3] = 0
    rgba[3:, :, 3] = 255
    path = tmp_path / "transparent_product.png"
    Image.fromarray(rgba, mode="RGBA").save(path)

    result = load_image(path)
    assert result.rgb_srgb.shape == (8, 10, 3)
    assert result.alpha_mask is not None
    assert np.all(result.alpha_mask[:3] == 0)
    assert np.all(result.alpha_mask[3:] == 1)
    assert result.metadata["alpha_preserved_separately"] is True


def test_image_loader_handles_rgb_photo(tmp_path):
    rgb = np.full((6, 7, 3), 128, dtype=np.uint8)
    path = tmp_path / "photo.jpg"
    Image.fromarray(rgb, mode="RGB").save(path, quality=100)

    result = load_image(path)
    assert result.rgb_srgb.shape == (6, 7, 3)
    assert result.alpha_mask is None
    assert result.metadata["format"] == "JPEG"
