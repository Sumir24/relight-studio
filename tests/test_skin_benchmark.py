from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.skin_benchmark import _save_gray_png, compute_roi_metrics, srgb_to_lab  # noqa: E402


def test_srgb_to_lab_neutral_and_hue_wrap():
    grey = np.full((8, 8, 3), 0.5, dtype=np.float32)
    lab = srgb_to_lab(grey)
    assert float(np.max(np.abs(lab[..., 1:]))) < 0.1
    assert 50.0 < float(np.mean(lab[..., 0])) < 56.0


def test_compute_roi_metrics_luminance_clip_lap_and_lab():
    h, w = 32, 32
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    tex = 0.04 * np.sin(xx * 1.5) * np.cos(yy * 1.5)
    orig = np.zeros((h, w, 3), dtype=np.float32)
    orig[..., 0] = np.clip(0.50 + tex, 0.0, 1.0)
    orig[..., 1] = np.clip(0.38 + tex, 0.0, 1.0)
    orig[..., 2] = np.clip(0.32 + tex, 0.0, 1.0)

    relit = np.clip(orig * 1.5, 0.0, 1.0)
    relit[:4, :4] = 0.99
    roi = np.ones((h, w), dtype=bool)

    m = compute_roi_metrics(orig, relit, roi)
    assert m["relit_mean_luma"] > m["orig_mean_luma"]
    assert m["relit_p99_luma"] >= m["relit_p95_luma"] >= m["relit_mean_luma"]
    assert m["relit_clip_pct"] > 0.0
    assert m["lap_ratio"] > 1.25
    assert -180.0 <= m["lab_delta_hue_deg"] <= 180.0
    assert m["lab_delta_e"] > 0.0


def test_save_gray_png_writes_valid_image(tmp_path: Path):
    arr = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    out = tmp_path / "debug.png"
    _save_gray_png(arr, out, scale_max=1.0)
    assert out.is_file()
    assert out.stat().st_size > 0

