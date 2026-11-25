#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight_engine.geometry import backproject_depth, estimate_normals
from relight_engine.adapters import (
    HuggingFaceDepthAnythingV2, DepthPrediction, SemanticPrediction, build_scene_buffers,
    HuggingFaceGroundedSAM2, Florence2SceneDiscovery, SemanticRoutingPolicy,
    camera_from_horizontal_fov,
)
from relight_engine.image_io import load_image
from relight_engine.scene_policy import semantic_key
from relight_engine.scene_domains import SceneDomainHint, SUPPORTED_QA_DOMAINS
from relight_engine.quality import evaluate_scene_buffers
from relight_engine.adapters.runtime import is_full_commit_sha, revision_descriptor


def save_gray(path: Path, array: np.ndarray) -> None:
    a = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(a)
    if not np.any(valid):
        raise ValueError(f"No finite values to visualize: {path}")
    lo, hi = np.quantile(a[valid], [0.02, 0.98])
    if hi - lo < 1e-8:
        lo = float(np.min(a[valid]))
        hi = float(np.max(a[valid]))
    if hi - lo < 1e-8:
        const_val = float(np.clip(lo, 0.0, 1.0)) if 0.0 <= lo <= 1.0 else 0.5
        vis = np.full(a.shape, int(round(const_val * 255)), dtype=np.uint8)
    else:
        vis = np.round(np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.round(np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def safe_filename(label: str) -> str:
    key = re.sub(r"[^a-z0-9_]+", "_", semantic_key(label)).strip("_") or "region"
    if len(key) <= 72:
        return key
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{key[:63].rstrip('_')}_{digest}"


def label_color(label: str) -> np.ndarray:
    # Stable pseudo-random palette from the label itself; no scene class mapping.
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(label))
    values = []
    for channel in range(3):
        seed = (seed * 1664525 + 1013904223 + channel * 7919) & 0xFFFFFFFF
        values.append(0.2 + 0.75 * ((seed >> 8) & 0xFFFF) / 65535.0)
    return np.asarray(values, dtype=np.float32)


def close_model_stage(stage: str, model_id: str, started: float, backend) -> dict:
    """Capture allocator counters and close an inference adapter safely."""
    before = {}
    cleanup = {}
    try:
        before = backend.runtime_snapshot()
    except Exception as exc:
        before = {"diagnostic_error": str(exc)}
    try:
        cleanup = backend.close()
    except Exception as exc:
        cleanup = {"cleanup_error": str(exc)}
    return {
        "stage": stage,
        "model_id": model_id,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "runtime_before_release": before,
        "cleanup": cleanup,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run domain-agnostic scene analysis. Auto-discover labels or pass image-specific prompts."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("scene_analysis"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--hfov", type=float, required=True,
                        help="Horizontal field of view. Supply calibrated/EXIF value or declare an approximation.")
    parser.add_argument("--hfov-source", default="assumed", choices=["assumed", "exif", "camera_profile", "checkerboard", "user_calibrated"])
    polarity = parser.add_mutually_exclusive_group(required=True)
    polarity.add_argument("--near-is-high", action="store_true",
                          help="Use only after verifying near/far polarity for the exact depth checkpoint.")
    polarity.add_argument("--far-is-high", action="store_true",
                          help="Use only after verifying near/far polarity for the exact depth checkpoint.")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Optional image-specific region/object labels. If omitted, Florence-2 discovers candidates.")
    parser.add_argument("--protected-labels", nargs="*", default=[],
                        help="Optional labels that must be protected from relighting; no classes are protected by default.")
    parser.add_argument("--lightable-labels", nargs="*", default=None,
                        help="Optional allow-list. Default: all visible pixels are candidates except explicit protected labels.")
    parser.add_argument("--box-threshold", type=float, default=0.28)
    parser.add_argument("--text-threshold", type=float, default=0.22)
    parser.add_argument("--max-detections", type=int, default=3)
    parser.add_argument("--discovery-model", default=Florence2SceneDiscovery.DEFAULT_MODEL)
    parser.add_argument("--scene-domain", default="auto", choices=["auto", *SUPPORTED_QA_DOMAINS],
                        help="Optional QA/profile hint only; it does not switch renderer physics.")
    parser.add_argument("--depth-model", default=HuggingFaceDepthAnythingV2.DEFAULT_RELATIVE_MODEL)
    parser.add_argument("--depth-revision", default=None,
                        help="Optional Hugging Face revision; use a full 40-character commit SHA for immutable pinning.")
    parser.add_argument("--discovery-revision", default=None,
                        help="Optional Florence-2 revision; use a full commit SHA to pin executed remote code.")
    parser.add_argument("--detector-revision", default=None,
                        help="Optional Grounding DINO revision; use a full 40-character commit SHA for immutable pinning.")
    parser.add_argument("--segmenter-revision", default=None,
                        help="Optional SAM2 revision; use a full 40-character commit SHA for immutable pinning.")
    parser.add_argument("--normal-backend", default="depth_derived",
                        help="Surface normal backend: 'depth_derived' or a registered normal backend (e.g. 'marigold_normals').")
    parser.add_argument("--normal-ensemble", type=int, default=3,
                        help="Ensemble size for diffusion-based normal backends (default: 3).")
    parser.add_argument("--normal-revision", default=None,
                        help="Optional Hugging Face revision for external normal backend.")
    parser.add_argument("--intrinsic-backend", default="none",
                        help="Optional intrinsic backend: 'none' or a registered backend (e.g. 'marigold_intrinsics_lighting').")
    parser.add_argument("--intrinsic-ensemble", type=int, default=3,
                        help="Ensemble size for diffusion-based intrinsic backends (default: 3).")
    parser.add_argument("--intrinsic-revision", default=None,
                        help="Optional Hugging Face revision for intrinsic backend.")
    parser.add_argument("--strict-model-revisions", action="store_true",
                        help="Require full 40-character commit SHAs for every model used in this run.")
    args = parser.parse_args()

    if args.strict_model_revisions:
        required_revisions = {
            "Depth Anything": args.depth_revision,
            "Grounding DINO": args.detector_revision,
            "SAM2": args.segmenter_revision,
        }
        if not args.prompts:
            required_revisions["Florence-2"] = args.discovery_revision
        if args.normal_backend != "depth_derived":
            required_revisions[f"Normals ({args.normal_backend})"] = args.normal_revision
        if args.intrinsic_backend != "none":
            required_revisions[f"Intrinsics ({args.intrinsic_backend})"] = args.intrinsic_revision
        missing = [
            model_name for model_name, revision in required_revisions.items()
            if not is_full_commit_sha(revision)
        ]
        if missing:
            parser.error(
                "--strict-model-revisions requires full 40-character commit SHAs for: "
                + ", ".join(missing)
            )

    if not args.image.is_file():
        parser.error(f"Input image does not exist: {args.image}")
    args.out.mkdir(parents=True, exist_ok=True)
    image_input = load_image(args.image)
    rgb = image_input.rgb_srgb
    h, w = rgb.shape[:2]
    policy = SemanticRoutingPolicy(
        protected_labels=tuple(args.protected_labels),
        lightable_labels=None if args.lightable_labels is None else tuple(args.lightable_labels),
    )
    policy.validate()
    domain_hint = SceneDomainHint(domain=args.scene_domain, source="user_hint" if args.scene_domain != "auto" else "not_classified")
    domain_hint.validate()
    inference_runtime: dict = {
        "device_requested": args.device,
        "model_revisions": {
            "depth": revision_descriptor(args.depth_revision),
            "discovery": (
                revision_descriptor(args.discovery_revision)
                if not args.prompts else {"revision": None, "revision_kind": "skipped_user_prompts"}
            ),
            "detector": revision_descriptor(args.detector_revision),
            "segmenter": revision_descriptor(args.segmenter_revision),
            "normals": (
                revision_descriptor(args.normal_revision)
                if args.normal_backend != "depth_derived"
                else {"revision": None, "revision_kind": "depth_derived_analytic"}
            ),
            "intrinsics": (
                revision_descriptor(args.intrinsic_revision)
                if args.intrinsic_backend != "none"
                else {"revision": None, "revision_kind": "disabled"}
            ),
        },
        "allocator_metric_scope": "PyTorch backend allocator counters; not total process/system memory",
        "stages": [],
    }

    depth_backend = HuggingFaceDepthAnythingV2(
        model_id=args.depth_model, output_kind="relative",
        relative_near_is_high=bool(args.near_is_high), device=args.device,
        revision=args.depth_revision,
    )
    print("[1/5] Estimating relative depth...")
    depth_started = perf_counter()
    try:
        depth: DepthPrediction = depth_backend.estimate(rgb)
    except Exception as exc:
        print(f"ERROR: Depth inference failed: {exc}", file=sys.stderr)
        print("Install requirements-models.txt and ensure the checkpoint is cached or downloadable.", file=sys.stderr)
        return 2
    finally:
        inference_runtime["stages"].append(
            close_model_stage("depth", args.depth_model, depth_started, depth_backend)
        )

    discovery_metadata: dict = {"mode": "user_prompts" if args.prompts else "automatic_open_vocabulary_discovery"}
    if args.prompts:
        inference_runtime["stages"].append({
            "stage": "scene_discovery",
            "status": "skipped_user_prompts_supplied",
            "elapsed_seconds": 0.0,
        })
        prompts = tuple(args.prompts)
        caption = ""
    else:
        print("[2/5] Discovering image-specific regions with Florence-2...")
        discovery_backend = Florence2SceneDiscovery(
            model_id=args.discovery_model, device=args.device,
            revision=args.discovery_revision,
        )
        discovery_started = perf_counter()
        try:
            discovery = discovery_backend.discover(rgb)
        except Exception as exc:
            print(f"ERROR: Automatic scene discovery failed: {exc}", file=sys.stderr)
            print("Supply --prompts with visible objects/regions, or install the model dependencies and checkpoints.", file=sys.stderr)
            return 2
        finally:
            inference_runtime["stages"].append(
                close_model_stage(
                    "scene_discovery", args.discovery_model,
                    discovery_started, discovery_backend
                )
            )
        prompts = discovery.prompts
        caption = discovery.caption
        discovery_metadata = {
            **discovery.metadata,
            "mode": "automatic_open_vocabulary_discovery",
            "caption": caption,
            "candidate_regions": discovery.regions,
        }

    print("[3/5] Building open-vocabulary region masks...")
    segmentation_backend = HuggingFaceGroundedSAM2(
        device=args.device, box_threshold=args.box_threshold,
        text_threshold=args.text_threshold, max_detections_per_prompt=args.max_detections,
        detector_revision=args.detector_revision,
        segmenter_revision=args.segmenter_revision,
    )
    segmentation_started = perf_counter()
    try:
        seg = segmentation_backend.estimate(rgb, prompts=prompts, policy=policy)
    except Exception as exc:
        print(f"ERROR: Semantic segmentation failed: {exc}", file=sys.stderr)
        print("No fixed class list or heuristic segmentation fallback was substituted.", file=sys.stderr)
        return 2
    finally:
        inference_runtime["stages"].append(
            close_model_stage(
                "segmentation", f"{segmentation_backend.detector_id} + {segmentation_backend.segmenter_id}",
                segmentation_started, segmentation_backend
            )
        )

    if float(np.mean(seg.lightable_mask > 0.05)) < 0.01:
        print("ERROR: The current semantic policy protects nearly the entire image; review label names.", file=sys.stderr)
        return 2

    print("[4/5] Building camera-aligned geometry and generic scene buffers...")
    calibration = camera_from_horizontal_fov(w, h, args.hfov, source=args.hfov_source)
    points = backproject_depth(depth.depth, calibration.camera)
    normals, normal_conf = estimate_normals(points, depth.depth, depth.confidence)
    geometry_conf = np.clip(normal_conf, 0, 1)
    geometry_conf_source = "edge-aware log-depth curvature and normal-validity heuristic; not calibrated"

    normal_meta: dict = {
        "backend": args.normal_backend,
        "source": "depth_derived_analytic" if args.normal_backend == "depth_derived" else "model_estimated",
    }
    if args.normal_backend != "depth_derived":
        from relight_engine.adapters.registry import create_normal_backend
        print(f"[4b/5] Estimating surface normals with {args.normal_backend}...")
        norm_kwargs = {"device": args.device, "ensemble_size": max(1, int(args.normal_ensemble))}
        if args.normal_revision:
            norm_kwargs["revision"] = args.normal_revision
        normal_backend = create_normal_backend(args.normal_backend, **norm_kwargs)
        norm_started = perf_counter()
        try:
            norm_pred = normal_backend.estimate(rgb, depth=depth.depth)
            normals = norm_pred.normals.astype(np.float32)
            geometry_conf = np.clip(norm_pred.confidence * np.where(depth.depth > 0, 1.0, 0.0), 0.0, 1.0).astype(np.float32)
            geometry_conf_source = str(norm_pred.metadata.get("confidence_source", f"normal backend {args.normal_backend}"))
            normal_meta = {
                "backend": args.normal_backend,
                **dict(norm_pred.metadata),
            }
        except Exception as exc:
            print(f"ERROR: Normal estimation failed: {exc}", file=sys.stderr)
            return 2
        finally:
            inference_runtime["stages"].append(
                close_model_stage("normals", args.normal_backend, norm_started, normal_backend)
            )

    intr_albedo = None
    intr_shading = None
    intr_residual = None
    intr_roughness = None
    intr_metalness = None
    intr_spec_level = None
    intr_meta: dict = {}
    mat_meta: dict = {}
    if args.intrinsic_backend != "none":
        from relight_engine.adapters.registry import create_intrinsic_backend
        print(f"[4c/5] Estimating intrinsic buffers with {args.intrinsic_backend}...")
        intr_kwargs = {"device": args.device, "ensemble_size": max(1, int(args.intrinsic_ensemble))}
        if args.intrinsic_revision:
            intr_kwargs["revision"] = args.intrinsic_revision
        intrinsic_backend = create_intrinsic_backend(args.intrinsic_backend, **intr_kwargs)
        intr_started = perf_counter()
        try:
            intr_pred = intrinsic_backend.estimate(rgb)
            intr_albedo = intr_pred.albedo
            intr_shading = intr_pred.source_shading
            intr_residual = intr_pred.non_diffuse_residual
            intr_roughness = intr_pred.roughness
            intr_metalness = intr_pred.metalness
            intr_spec_level = intr_pred.specular_level
            intr_meta = dict(intr_pred.metadata)
            for mname, marr in (("roughness", intr_roughness), ("metalness", intr_metalness), ("specular_level", intr_spec_level)):
                if marr is not None:
                    mat_meta[mname] = {"source": "model_estimated", "calibrated": False}
        except Exception as exc:
            print(f"ERROR: Intrinsic decomposition failed: {exc}", file=sys.stderr)
            return 2
        finally:
            inference_runtime["stages"].append(
                close_model_stage("intrinsics", args.intrinsic_backend, intr_started, intrinsic_backend)
            )

    semantics = SemanticPrediction(
        lightable_mask=seg.lightable_mask, protected_mask=seg.protected_mask,
        class_masks=seg.masks, instance_masks=seg.instance_masks, metadata=seg.metadata,
        roughness=intr_roughness, metalness=intr_metalness, specular_level=intr_spec_level,
        material_metadata=mat_meta,
        albedo=intr_albedo, source_shading=intr_shading, non_diffuse_residual=intr_residual,
        intrinsic_metadata=intr_meta,
    )
    scene = build_scene_buffers(rgb, depth, semantics, alpha_mask=image_input.alpha_mask)
    scene.geometry_confidence = geometry_conf.astype(np.float32)
    scene.normals = normals.astype(np.float32)
    scene.metadata["geometry_confidence_source"] = geometry_conf_source
    scene.metadata["geometry_confidence_calibrated"] = False
    scene.validate()

    print("[5/5] Saving generic scene buffers, masks and provenance...")
    save_gray(args.out / "depth_relative.png", depth.depth)
    Image.fromarray(np.round(np.clip((normals + 1.0) * 0.5, 0, 1) * 255).astype(np.uint8), mode="RGB").save(args.out / "normals_camera_space.png")
    save_mask(args.out / "geometry_confidence.png", geometry_conf)
    save_mask(args.out / "mask_lightable.png", seg.lightable_mask)
    save_mask(args.out / "mask_protected.png", seg.protected_mask)

    overlay = rgb.copy()
    for label, mask in seg.masks.items():
        save_mask(args.out / f"mask_{safe_filename(label)}.png", mask)
        alpha = (np.clip(mask, 0, 1) * 0.42)[..., None]
        overlay = overlay * (1 - alpha) + label_color(label) * alpha
    for instance_id, mask in seg.instance_masks.items():
        save_mask(args.out / f"instance_mask_{safe_filename(instance_id)}.png", mask)
    Image.fromarray(np.round(np.clip(overlay, 0, 1) * 255).astype(np.uint8), mode="RGB").save(args.out / "semantic_overlay.jpg", quality=94)

    buffers = {
        "rgb_srgb": scene.rgb_srgb,
        "depth": scene.depth,
        "depth_confidence": scene.depth_confidence,
        "geometry_confidence": scene.geometry_confidence,
        "normals": scene.normals,
        "lightable_mask": scene.lightable_mask,
        "protected_mask": seg.protected_mask,
    }
    if image_input.alpha_mask is not None:
        buffers["alpha_mask"] = image_input.alpha_mask
    if scene.albedo is not None:
        buffers["albedo"] = scene.albedo
        Image.fromarray(np.round(np.clip(scene.albedo, 0, 1) * 255).astype(np.uint8), mode="RGB").save(args.out / "intrinsic_albedo.png")
    if scene.source_shading is not None:
        buffers["source_shading"] = scene.source_shading
        sh_vis = scene.source_shading if scene.source_shading.ndim == 2 else np.mean(scene.source_shading, axis=-1)
        save_gray(args.out / "intrinsic_shading.png", sh_vis)
    if scene.non_diffuse_residual is not None:
        buffers["non_diffuse_residual"] = scene.non_diffuse_residual
    for mname in ("roughness", "metalness", "specular_level"):
        mval = getattr(scene, mname)
        if mval is not None:
            buffers[mname] = mval
    buffers.update({f"mask_{safe_filename(k)}": v for k, v in seg.masks.items()})
    buffers.update({f"mask_instance_{safe_filename(k)}": v for k, v in seg.instance_masks.items()})
    np.savez_compressed(args.out / "scene_buffers.npz", **buffers)

    # Structural QA catches degenerate model outputs but does not claim ground-truth accuracy.
    quality_report = evaluate_scene_buffers(scene)
    (args.out / "scene_quality_report.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )

    manifest = {
        "status": "generic_scene_analysis_complete_not_relighting",
        "image": str(args.image.resolve()),
        "image_input": image_input.metadata,
        "resolution": [w, h],
        "scene_domain_hint": domain_hint.to_metadata(),
        "scene_discovery": discovery_metadata,
        "semantic_labels": sorted(seg.masks.keys()),
        "instance_masks": sorted(seg.instance_masks.keys()),
        "depth": depth.metadata,
        "normals": normal_meta,
        "intrinsic_metadata": scene.intrinsic_metadata,
        "material_metadata": scene.material_metadata,
        "geometry_confidence": {
            "source": scene.metadata["geometry_confidence_source"],
            "calibrated": False,
        },
        "segmentation": seg.metadata,
        "inference_runtime": inference_runtime,
        "detections": [d.__dict__ for d in seg.detections],
        "semantic_policy": policy.to_metadata(),
        "camera": {
            "fx": calibration.camera.fx, "fy": calibration.camera.fy,
            "cx": calibration.camera.cx, "cy": calibration.camera.cy,
            "source": calibration.source,
            "horizontal_fov_degrees": calibration.horizontal_fov_degrees,
            "calibrated": calibration.calibrated,
            "note": calibration.note,
        },
        "scene_quality_report": "scene_quality_report.json",
        "structural_quality_status": quality_report["status"],
        "lighting_readiness": quality_report["lighting_readiness"],
        "metrics": {
            "lightable_fraction": float(np.mean(seg.lightable_mask > 0.05)),
            "protected_fraction": float(np.mean(seg.protected_mask > 0.05)),
            "valid_depth_fraction": float(np.mean(depth.depth > 0)),
            "geometry_confidence_mean": float(np.mean(geometry_conf)),
            "detections_count": len(seg.detections),
        },
        "material_estimation": (
            f"model-estimated rough/metal/specular priors from {args.intrinsic_backend} (uncalibrated)"
            if mat_meta
            else "not implemented; no physical material labels are inferred by segmentation"
        ),
        "source_light_estimation": "not implemented; requires user calibration or a separately validated estimator",
        "limitations": [
            "Depth is relative, not metric; no physical distance claims are valid.",
            "Camera intrinsics are FOV-derived unless calibrated metadata was supplied.",
            "Open-vocabulary region discovery is a candidate list, not exhaustive semantic ground truth.",
            "Detector and mask-quality scores are not calibrated probabilities.",
            *(
                ["Material maps from intrinsic appearance inference are uncalibrated; original light remains user-calibrated."]
                if mat_meta
                else ["No material maps or original-light estimate are produced by this analysis pass."]
            ),
            "All image pixels are candidates by default; only explicit protected labels are excluded.",
            "No relit output is generated by this scene-analysis command.",
        ],
    }
    (args.out / "scene_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["metrics"], indent=2))
    print(f"Discovered/used labels: {', '.join(prompts)}")
    print(f"Saved scene analysis to: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
