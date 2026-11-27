#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relight_engine.workflows import run_scene_relight


def main() -> int:
    parser = argparse.ArgumentParser(description="Relight an analyzed scene using explicit source and target calibrations.")
    parser.add_argument("--scene-dir", required=True, type=Path, help="Directory produced by run_scene_analysis.py")
    parser.add_argument("--config", required=True, type=Path, help="User-reviewed JSON lighting/material calibration")
    parser.add_argument("--output", type=Path, default=Path("relight_output.png"))
    parser.add_argument("--renderer", default=None, choices=["auto", "ratio", "intrinsic", "hybrid_neural"],
                        help="Override settings.renderer_mode.")
    parser.add_argument("--relighter", default=None,
                        help="Registered neural relighter backend for hybrid_neural mode (e.g. ic_light).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    try:
        report = run_scene_relight(
            args.scene_dir, args.config, args.output,
            renderer_mode=args.renderer, relighter_backend=args.relighter, device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["engine_metrics"], indent=2))
    print(f"Saved render: {args.output.resolve()}")
    print(f"Saved report: {Path(str(args.output) + '.report.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
