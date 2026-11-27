#!/usr/bin/env python3
"""Run structural QA on a saved scene_buffers.npz from run_scene_analysis.py."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight_engine.contracts import SceneBuffers
from relight_engine.quality import evaluate_scene_buffers


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate depth, geometry and masks without assuming a scene category.")
    parser.add_argument("--scene-npz", required=True, type=Path, help="scene_buffers.npz")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional scene_manifest.json")
    parser.add_argument("--out", type=Path, default=Path("scene_quality_report.json"))
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()

    manifest = {}
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    depth_kind = manifest.get("depth", {}).get("depth_kind", "relative")
    with np.load(args.scene_npz, allow_pickle=False) as data:
        required = {"rgb_srgb", "depth", "depth_confidence", "lightable_mask"}
        missing = required - set(data.files)
        if missing:
            raise SystemExit(f"Missing required scene arrays: {sorted(missing)}")
        arrays = {key: data[key] for key in data.files}

    semantic_masks = {
        key[len("mask_"):]: value for key, value in arrays.items()
        if key.startswith("mask_") and not key.startswith("mask_instance_")
        and key not in ("mask_lightable", "mask_protected")
    }
    instance_masks = {
        key[len("mask_instance_"):]: value for key, value in arrays.items()
        if key.startswith("mask_instance_")
    }
    scene = SceneBuffers(
        rgb_srgb=arrays["rgb_srgb"], depth=arrays["depth"],
        depth_confidence=arrays["depth_confidence"], lightable_mask=arrays["lightable_mask"],
        depth_kind=depth_kind,
        normals=arrays.get("normals"), geometry_confidence=arrays.get("geometry_confidence"),
        alpha_mask=arrays.get("alpha_mask"), protected_mask=arrays.get("protected_mask"),
        semantic_masks=semantic_masks, instance_masks=instance_masks,
        roughness=arrays.get("roughness"),
        metalness=arrays.get("metalness"),
        specular_level=arrays.get("specular_level"),
        specular_color=arrays.get("specular_color"),
        material_confidence=arrays.get("material_confidence"),
        material_metadata=manifest.get("material_metadata", {}),
        metadata={
            "source_manifest": str(args.manifest.resolve()) if args.manifest else None,
            "source_npz": str(args.scene_npz.resolve()),
            "depth": manifest.get("depth", {}),
            "geometry_confidence_source": manifest.get("geometry_confidence", {}).get("source"),
            "material_maps_estimated": False,
        },
    )
    report = evaluate_scene_buffers(scene)
    report["source_manifest_status"] = manifest.get("status")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "warnings": report["warnings"], "out": str(args.out.resolve())}, indent=2))
    return 2 if args.fail_on_warnings and report["warnings"] else (1 if report["errors"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
