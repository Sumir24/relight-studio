#!/usr/bin/env python3
"""CPU integration smoke test for the real Hugging Face scene-analysis checkpoints.

This script is intentionally separate from unit tests: it downloads real model
weights and a public example photograph, then exercises actual inference. It is
designed for GitHub Actions or a developer machine with network access. It does
not validate Apple MPS behaviour or visual-quality ground truth.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image

from relight_engine.adapters import (
    Florence2SceneDiscovery,
    HuggingFaceDepthAnythingV2,
    HuggingFaceGroundedSAM2,
)

FIXTURE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/transformers/tasks/car.jpg?download=true"
)
OUTPUT_PATH = Path("model_smoke_report.json")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_fixture() -> np.ndarray:
    request = urllib.request.Request(
        FIXTURE_URL,
        headers={"User-Agent": "Relight-Studio-model-smoke/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        image = Image.open(response).convert("RGB")
    # Bound CPU integration-test cost while retaining an actual natural image.
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def timed_stage(name: str, fn):
    started = time.perf_counter()
    print(f"[START] {name}", flush=True)
    result = fn()
    elapsed = round(time.perf_counter() - started, 3)
    print(f"[PASS] {name}: {elapsed:.3f}s", flush=True)
    return result, elapsed


def main() -> int:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rgb = load_fixture()
    height, width = rgb.shape[:2]
    report: dict = {
        "status": "running",
        "device": "cpu",
        "python_torch": torch.__version__,
        "transformers": transformers.__version__,
        "image": {
            "source": FIXTURE_URL,
            "width": width,
            "height": height,
        },
        "stages": [],
        "limitations": [
            "This is real CPU checkpoint inference on GitHub Actions, not Apple MPS validation.",
            "Successful inference does not establish depth, mask, normal, material, or relighting ground-truth accuracy.",
            "The fixture is a car photograph; this smoke test is not a multi-domain benchmark.",
        ],
    }

    # 1. Depth Anything V2: test actual processor/model loading and raw inference.
    depth_adapter = HuggingFaceDepthAnythingV2(
        device="cpu",
        relative_near_is_high=True,  # Only used by estimate(); polarity is not evaluated here.
    )
    try:
        (raw_depth, depth_metadata), elapsed = timed_stage(
            "Depth Anything V2 raw inference",
            lambda: depth_adapter.predict_raw(rgb),
        )
        if raw_depth.shape != (height, width) or not np.isfinite(raw_depth).all():
            raise RuntimeError("Depth Anything V2 returned an invalid raw depth map.")
        report["stages"].append({
            "name": "depth",
            "status": "passed",
            "elapsed_seconds": elapsed,
            "model_id": depth_metadata["model_id"],
            "model_revision": {
                "revision": depth_metadata.get("revision"),
                "revision_kind": depth_metadata.get("revision_kind"),
                "immutable_commit_pinned": depth_metadata.get("immutable_commit_pinned"),
            },
            "output_shape": list(raw_depth.shape),
            "raw_range": [
                float(np.min(raw_depth)),
                float(np.max(raw_depth)),
            ],
        })
    finally:
        depth_adapter.close()
        del depth_adapter
        gc.collect()

    # 2. Florence-2: exercise both detailed caption and dense-region discovery.
    discovery_adapter = Florence2SceneDiscovery(
        device="cpu",
        max_new_tokens=256,
        max_prompts=16,
    )
    try:
        discovery, elapsed = timed_stage(
            "Florence-2 scene discovery",
            lambda: discovery_adapter.discover(rgb),
        )
        if not discovery.prompts:
            raise RuntimeError("Florence-2 returned no region prompts.")
        report["stages"].append({
            "name": "florence2",
            "status": "passed",
            "elapsed_seconds": elapsed,
            "model_id": discovery.metadata["model_id"],
            "model_revision": {
                "revision": discovery.metadata.get("revision"),
                "revision_kind": discovery.metadata.get("revision_kind"),
                "immutable_commit_pinned": discovery.metadata.get("immutable_commit_pinned"),
            },
            "remote_code_revision_pinned": discovery.metadata.get("remote_code_revision_pinned"),
            "prompt_count": len(discovery.prompts),
            "prompts": list(discovery.prompts),
            "caption_present": bool(discovery.caption),
        })
        discovered_prompts = list(discovery.prompts)
    finally:
        discovery_adapter.close()
        del discovery_adapter
        gc.collect()

    # 3. Grounding DINO + SAM2: use the documented car fixture label alongside
    # Florence's discovered labels. This label belongs only to this smoke fixture,
    # not to the universal runtime's scene taxonomy.
    prompts = list(dict.fromkeys(["car", *discovered_prompts]))[:16]
    segmentation_adapter = HuggingFaceGroundedSAM2(
        device="cpu",
        box_threshold=0.20,
        text_threshold=0.15,
        max_detections_per_prompt=2,
        min_area_fraction=0.0005,
    )
    try:
        segmentation, elapsed = timed_stage(
            "Grounding DINO + SAM2 segmentation",
            lambda: segmentation_adapter.estimate(rgb, prompts=prompts),
        )
        lifecycle = segmentation.metadata.get("model_lifecycle", {})
        if not segmentation.detections or not segmentation.instance_masks:
            raise RuntimeError(
                "Grounding DINO returned no candidate accepted by SAM2 for the car fixture. "
                "Inspect detections and thresholds; the smoke test does not fabricate boxes."
            )
        if lifecycle.get("models_loaded_sequentially") is not True:
            raise RuntimeError("Segmentation did not report sequential model execution.")
        if lifecycle.get("max_accelerator_models_loaded_concurrently") != 1:
            raise RuntimeError("Unexpected simultaneous detector/segmenter residency.")
        for instance_id, mask in segmentation.instance_masks.items():
            if mask.shape != (height, width) or not np.isfinite(mask).all():
                raise RuntimeError(f"Invalid SAM2 instance mask: {instance_id}")
        report["stages"].append({
            "name": "grounding_dino_sam2",
            "status": "passed",
            "elapsed_seconds": elapsed,
            "detector_id": segmentation.metadata["detector_id"],
            "detector_revision": segmentation.metadata["detector_model_revision"],
            "segmenter_id": segmentation.metadata["segmenter_id"],
            "segmenter_revision": segmentation.metadata["segmenter_model_revision"],
            "detection_count": len(segmentation.detections),
            "instance_mask_count": len(segmentation.instance_masks),
            "lifecycle": lifecycle,
        })
    finally:
        segmentation_adapter.close()
        del segmentation_adapter
        gc.collect()

    report["status"] = "passed"
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved report: {OUTPUT_PATH.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
