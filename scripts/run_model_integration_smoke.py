#!/usr/bin/env python3
"""Run real Hugging Face checkpoint inference stages independently on CPU.

A stage failure is recorded and does not prevent later independent model stages
from running. The command still exits non-zero if any stage fails. It downloads
real weights and a public fixture photograph; it is not an MPS or quality benchmark.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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

# Immutable Hugging Face revisions recorded in LICENSE_MODEL_NOTES.md.
DEPTH_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
FLORENCE_REVISION = "15bd11834572457e1e8e8fec7f25c83283902db2"
GROUNDING_DINO_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
SAM2_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"


def load_fixture() -> np.ndarray:
    request = urllib.request.Request(
        FIXTURE_URL,
        headers={"User-Agent": "Relight-Studio-model-smoke/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        image = Image.open(response).convert("RGB")
    image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def timed_stage(name: str, fn):
    started = time.perf_counter()
    print(f"[START] {name}", flush=True)
    result = fn()
    elapsed = round(time.perf_counter() - started, 3)
    print(f"[PASS] {name}: {elapsed:.3f}s", flush=True)
    return result, elapsed


def failed_stage(report: dict, name: str, exc: Exception, elapsed: float) -> None:
    message = f"{type(exc).__name__}: {exc}"
    print(f"[FAIL] {name}: {message}", file=sys.stderr, flush=True)
    report["stages"].append({
        "name": name,
        "status": "failed",
        "elapsed_seconds": round(elapsed, 3),
        "error_type": type(exc).__name__,
        "error": str(exc),
    })


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
        "model_revisions": {
            "depth": DEPTH_REVISION,
            "florence2": FLORENCE_REVISION,
            "grounding_dino": GROUNDING_DINO_REVISION,
            "sam2": SAM2_REVISION,
        },
        "stages": [],
        "limitations": [
            "This is real CPU checkpoint inference on GitHub Actions, not Apple MPS validation.",
            "Successful inference does not establish depth, mask, normal, material, or relighting ground-truth accuracy.",
            "The fixture is a car photograph; this smoke test is not a multi-domain benchmark.",
            "The fixture-specific fallback prompt 'car' is used only to test the known public car fixture when Florence discovery fails; production prompts remain image-specific.",
        ],
    }

    # 1. Depth Anything V2, independent raw-depth checkpoint check.
    depth_adapter = HuggingFaceDepthAnythingV2(
        device="cpu",
        relative_near_is_high=True,  # Polarity is not assessed in this smoke test.
        revision=DEPTH_REVISION,
    )
    started = time.perf_counter()
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
            "model_revision": depth_metadata,
            "output_shape": list(raw_depth.shape),
            "raw_range": [float(np.min(raw_depth)), float(np.max(raw_depth))],
        })
    except Exception as exc:
        failed_stage(report, "depth", exc, time.perf_counter() - started)
    finally:
        depth_adapter.close()
        del depth_adapter
        gc.collect()

    # 2. Florence-2. Continue to segmentation using the known fixture label if it fails.
    discovered_prompts: list[str] = []
    discovery_adapter = Florence2SceneDiscovery(
        device="cpu",
        max_new_tokens=256,
        max_prompts=16,
        revision=FLORENCE_REVISION,
    )
    started = time.perf_counter()
    try:
        (discovery, elapsed) = timed_stage(
            "Florence-2 scene discovery",
            lambda: discovery_adapter.discover(rgb),
        )
        if not discovery.prompts:
            raise RuntimeError("Florence-2 returned no region prompts.")
        discovered_prompts = list(discovery.prompts)
        report["stages"].append({
            "name": "florence2",
            "status": "passed",
            "elapsed_seconds": elapsed,
            "model_id": discovery.metadata["model_id"],
            "model_revision": discovery.metadata,
            "prompt_count": len(discovery.prompts),
            "prompts": discovered_prompts,
            "caption_present": bool(discovery.caption),
        })
    except Exception as exc:
        failed_stage(report, "florence2", exc, time.perf_counter() - started)
    finally:
        discovery_adapter.close()
        del discovery_adapter
        gc.collect()

    # 3. Grounding DINO + SAM2, independently tested even if Florence failed.
    # "car" is an explicit label for this known test fixture only, not a production taxonomy.
    prompts = list(dict.fromkeys(["car", *discovered_prompts]))[:16]
    segmentation_adapter = HuggingFaceGroundedSAM2(
        device="cpu",
        box_threshold=0.20,
        text_threshold=0.15,
        max_detections_per_prompt=2,
        min_area_fraction=0.0005,
        detector_revision=GROUNDING_DINO_REVISION,
        segmenter_revision=SAM2_REVISION,
    )
    started = time.perf_counter()
    try:
        (segmentation, elapsed) = timed_stage(
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
            "prompt_source": "fixture_label_plus_florence_discovery" if discovered_prompts else "known_fixture_label_only",
            "detection_count": len(segmentation.detections),
            "instance_mask_count": len(segmentation.instance_masks),
            "lifecycle": lifecycle,
        })
    except Exception as exc:
        failed_stage(report, "grounding_dino_sam2", exc, time.perf_counter() - started)
    finally:
        segmentation_adapter.close()
        del segmentation_adapter
        gc.collect()

    failures = [stage for stage in report["stages"] if stage["status"] != "passed"]
    report["status"] = "passed" if not failures else "failed"
    report["passed_stage_count"] = len(report["stages"]) - len(failures)
    report["failed_stage_count"] = len(failures)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved report: {OUTPUT_PATH.resolve()}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
