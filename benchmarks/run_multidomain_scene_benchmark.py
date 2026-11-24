#!/usr/bin/env python3
"""Run the same neural scene-analysis pipeline over arbitrary image categories.

Each --case label is report metadata only. It is not passed to the detector or
renderer and does not select category-specific processing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE_SCRIPT = ROOT / "scripts" / "run_scene_analysis.py"


def safe_key(value: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_-")
    if not key:
        raise ValueError("Case label must contain at least one letter or number.")
    return key[:64]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_case(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=PATH, got {raw!r}.")
    label, path_text = raw.split("=", 1)
    label = label.strip()
    path = Path(path_text).expanduser().resolve()
    if not label or not path.is_file():
        raise ValueError(f"Invalid case {raw!r}: label must be non-empty and file must exist.")
    return label, path


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one shared scene-analysis pipeline across mixed-domain images.")
    parser.add_argument("--case", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--hfov", type=float, default=60.0,
                        help="FOV assumption if no calibrated camera profile is supplied (default 60 degrees).")
    parser.add_argument("--hfov-source", default="assumed",
                        choices=["assumed", "exif", "camera_profile", "checkerboard", "user_calibrated"])
    polarity = parser.add_mutually_exclusive_group(required=True)
    polarity.add_argument("--near-is-high", action="store_true")
    polarity.add_argument("--far-is-high", action="store_true")
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="Optional shared manual prompts; omit to use per-image automatic discovery.")
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()

    try:
        cases = [parse_case(item) for item in args.case]
    except ValueError as exc:
        parser.error(str(exc))
    keys = [safe_key(label) for label, _ in cases]
    if len(set(keys)) != len(keys):
        parser.error("Case labels collide after output-name normalization.")

    output_root = args.out.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    warning_count = 0
    for (label, image_path), key in zip(cases, keys):
        case_out = output_root / key
        command = [
            sys.executable, str(SCENE_SCRIPT), "--image", str(image_path),
            "--out", str(case_out), "--device", args.device,
            "--hfov", str(args.hfov), "--hfov-source", args.hfov_source,
        ]
        command.append("--near-is-high" if args.near_is_high else "--far-is-high")
        if args.prompts:
            command.extend(["--prompts", *args.prompts])
        completed = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
        manifest_path = case_out / "scene_manifest.json"
        quality_path = case_out / "scene_quality_report.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
        warnings = quality.get("warnings", [])
        warning_count += len(warnings)
        results.append({
            "label": label,
            "image": image_path.name,
            "sha256": sha256_file(image_path),
            "output_directory": str(case_out),
            "process_exit_code": completed.returncode,
            "scene_analysis_status": manifest.get("status", "failed_or_incomplete"),
            "structural_quality_status": quality.get("status", "not_available"),
            "domain_hint": manifest.get("scene_domain_hint"),
            "metrics": manifest.get("metrics", {}),
            "depth_model": manifest.get("depth", {}).get("model_id"),
            "discovery_model": manifest.get("scene_discovery", {}).get("model_id"),
            "semantic_labels": manifest.get("semantic_labels", []),
            "instance_count": len(manifest.get("instance_masks", [])),
            "warnings": warnings,
            "stderr_tail": completed.stderr[-3000:],
            "stdout_tail": completed.stdout[-1000:],
        })

    succeeded = sum(item["process_exit_code"] == 0 for item in results)
    status = "passed" if succeeded == len(results) and not (args.fail_on_warnings and warning_count) else (
        "partial" if succeeded else "failed"
    )
    report = {
        "schema_version": "1.0",
        "benchmark": "Relight Studio multi-domain scene understanding and geometry preflight",
        "status": status,
        "case_count": len(results),
        "successful_scene_analyses": succeeded,
        "warning_count": warning_count,
        "labels_affect_processing": False,
        "shared_scene_analysis_pipeline": str(SCENE_SCRIPT),
        "camera_fov": {"degrees": args.hfov, "source": args.hfov_source,
                       "is_calibrated": args.hfov_source in ("checkerboard", "user_calibrated")},
        "depth_polarity_assumption": "near_is_high" if args.near_is_high else "far_is_high",
        "cases": results,
        "non_claims": [
            "Case labels do not alter scene analysis or rendering.",
            "Structural scene QA is not a ground-truth accuracy metric.",
            "Relative depth is not metric distance.",
            "No final relighting-quality score is computed by this benchmark.",
        ],
    }
    report_path = output_root / "multidomain_scene_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "successful_scene_analyses": succeeded,
                      "case_count": len(results), "report": str(report_path)}, indent=2))
    if succeeded != len(results):
        return 1
    if args.fail_on_warnings and warning_count:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
