#!/usr/bin/env python3
"""Domain-agnostic input preflight for a list of arbitrary photographs.

Pass cases as --case LABEL=PATH. LABEL is a QA note only; it never changes processing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relight_engine.image_io import load_image


def summarize_case(label: str, path: Path) -> dict:
    digest_builder = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    loaded = load_image(path)
    rgb = loaded.rgb_srgb
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    h, w = rgb.shape[:2]
    alpha = loaded.alpha_mask
    return {
        "label": label,
        "source_file": path.name,
        "sha256": digest,
        "status": "decoded_and_validated",
        "format": loaded.metadata.get("format"),
        "original_mode": loaded.metadata.get("original_mode"),
        "width": int(w), "height": int(h),
        "aspect_ratio": float(w / h),
        "megapixels": float(w * h / 1_000_000),
        "rgb_range": [float(rgb.min()), float(rgb.max())],
        "channel_std": [float(x) for x in np.std(rgb, axis=(0, 1))],
        "luminance_percentiles": [float(x) for x in np.percentile(luminance, [1, 50, 99])],
        "near_black_fraction": float(np.mean(luminance < 0.01)),
        "near_white_fraction": float(np.mean(luminance > 0.99)),
        "alpha_present": alpha is not None,
        "transparent_fraction": float(np.mean(alpha < 0.99)) if alpha is not None else 0.0,
        "image_io_metadata": loaded.metadata,
        "scope": "input_decode_color_orientation_alpha_only_no_scene_or_relighting_validation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight mixed-domain photo inputs; labels are report metadata only.")
    parser.add_argument("--case", action="append", required=True, metavar="LABEL=PATH",
                        help="Repeat for each image. Any descriptive label is allowed.")
    parser.add_argument("--out", type=Path, default=Path("input_preflight_report.json"))
    parser.add_argument("--fixture-note", default=None,
                        help="Optional provenance note for temporary or curated benchmark fixtures.")
    args = parser.parse_args()
    cases = []
    failures = []
    for item in args.case:
        if "=" not in item:
            failures.append({"input": item, "error": "Expected LABEL=PATH."})
            continue
        label, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser()
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            cases.append(summarize_case(label.strip(), path))
        except Exception as exc:
            failures.append({"input": item, "error": f"{type(exc).__name__}: {exc}"})
    report = {
        "schema_version": "1.0",
        "benchmark": "Relight Studio arbitrary-image input preflight",
        "status": "passed" if cases and not failures else ("partial" if cases else "failed"),
        "case_count": len(cases), "failure_count": len(failures),
        "labels_affect_processing": False,
        "fixture_note": args.fixture_note,
        "cases": cases, "failures": failures,
        "non_claims": [
            "Successful decoding does not establish successful depth or segmentation inference.",
            "This preflight does not evaluate geometry, masks, materials, source lighting or rendered quality.",
            "Labels are descriptive QA metadata and do not select a renderer branch.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "case_count": len(cases), "failures": failures,
                      "out": str(args.out.resolve())}, indent=2))
    return 1 if failures or not cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
