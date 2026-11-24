#!/usr/bin/env python3
"""Run one depth inference and save both possible near/far interpretations for review."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight_engine.adapters import HuggingFaceDepthAnythingV2, relative_depth_polarity_candidates


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_relative_depth(path: Path, depth: np.ndarray) -> None:
    # The common range makes the two candidate previews directly comparable.
    normalized = np.clip((depth - 1.0) / 4.0, 0.0, 1.0)
    Image.fromarray(np.round(normalized * 255).astype(np.uint8), mode="L").save(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save both possible Depth Anything V2 relative-depth polarity interpretations."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("depth_polarity_review"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Input image does not exist: {args.image}")
    args.out.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb(args.image)
    # The configured polarity is intentionally not used here. We inspect the raw
    # checkpoint scores and render both possible mappings from the same inference.
    backend = HuggingFaceDepthAnythingV2(relative_near_is_high=True, device=args.device)
    try:
        raw_score, metadata = backend.predict_raw(rgb)
    except Exception as exc:
        print(f"ERROR: depth inference failed: {exc}", file=sys.stderr)
        print("Install requirements-models.txt and ensure the model checkpoint is available.", file=sys.stderr)
        return 2
    candidates = relative_depth_polarity_candidates(raw_score)
    save_relative_depth(args.out / "candidate_near_is_high.png", candidates["near_is_high"])
    save_relative_depth(args.out / "candidate_far_is_high.png", candidates["far_is_high"])
    source = Image.open(args.image).convert("RGB")
    near = Image.open(args.out / "candidate_near_is_high.png").convert("L").convert("RGB")
    far = Image.open(args.out / "candidate_far_is_high.png").convert("L").convert("RGB")
    panel_w = source.width
    panel_h = source.height
    board = Image.new("RGB", (panel_w * 3, panel_h + 44), "white")
    draw = ImageDraw.Draw(board)
    board.paste(source, (0, 44))
    board.paste(near, (panel_w, 44))
    board.paste(far, (panel_w * 2, 44))
    draw.text((12, 14), "Original", fill="black")
    draw.text((panel_w + 12, 14), "Assumption: raw score HIGH = near", fill="black")
    draw.text((panel_w * 2 + 12, 14), "Assumption: raw score LOW = near", fill="black")
    board.save(args.out / "depth_polarity_review.jpg", quality=94)
    print(f"Saved two candidate maps and comparison board to: {args.out.resolve()}")
    print("Review a known near object against a known farther region in this exact image.")
    print("Choose --near-is-high or --far-is-high for run_scene_analysis.py only after review.")
    print(f"Checkpoint: {metadata.get('model_id')}")
    print(f"Device: {metadata.get('device')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
