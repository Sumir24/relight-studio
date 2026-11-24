"""Structural QA for estimated scene buffers.

These checks detect malformed or degenerate model outputs. They do not establish
semantic correctness or perceptual relighting quality; those require annotated
real-photo benchmarks and human review.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any
import numpy as np

from .contracts import SceneBuffers


def _percentile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def _mask_stats(mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(mask, dtype=np.float32)
    coverage = values > 0.5
    return {
        "mean_soft_coverage": float(np.mean(values)),
        "binary_coverage_fraction": float(np.mean(coverage)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def evaluate_scene_buffers(scene: SceneBuffers) -> dict[str, Any]:
    """Validate buffer structure and calculate non-ground-truth QA diagnostics."""
    scene.validate()
    h, w = scene.shape
    visible = np.ones((h, w), dtype=bool)
    if scene.alpha_mask is not None:
        visible &= scene.alpha_mask > 0.01

    depth = np.asarray(scene.depth, dtype=np.float32)
    valid_depth = (depth > 0) & visible & np.isfinite(depth)
    depth_values = depth[valid_depth]
    depth_p02 = _percentile(depth_values, 2)
    depth_p50 = _percentile(depth_values, 50)
    depth_p98 = _percentile(depth_values, 98)
    depth_spread = None
    if depth_p02 is not None and depth_p50 is not None and depth_p98 is not None:
        depth_spread = float((depth_p98 - depth_p02) / max(abs(depth_p50), 1e-6))

    # Quantile bins are used instead of unique() to avoid a large memory spike on
    # high-resolution floating-point maps.
    depth_bin_count = 0
    if depth_values.size:
        lo, hi = np.percentile(depth_values, [1, 99])
        if hi > lo:
            depth_bin_count = int(np.unique(np.clip(
                np.floor((depth_values - lo) / (hi - lo) * 63), 0, 63
            ).astype(np.uint8)).size)

    depth_grad_p95 = None
    if np.count_nonzero(valid_depth) >= 16:
        # Neighbour differences are calculated only where both pixels are valid.
        dx_valid = valid_depth[:, 1:] & valid_depth[:, :-1]
        dy_valid = valid_depth[1:, :] & valid_depth[:-1, :]
        dx = np.abs(np.diff(np.log(np.maximum(depth, 1e-6)), axis=1))[dx_valid]
        dy = np.abs(np.diff(np.log(np.maximum(depth, 1e-6)), axis=0))[dy_valid]
        grads = np.concatenate([dx, dy]) if dx.size or dy.size else np.array([], np.float32)
        depth_grad_p95 = _percentile(grads, 95)

    confidence = scene.depth_confidence[visible]
    geometry_conf = None if scene.geometry_confidence is None else scene.geometry_confidence[visible]
    lightable = scene.lightable_mask[visible]
    protected = None if scene.protected_mask is None else scene.protected_mask[visible]

    normals_report: dict[str, Any] | None = None
    if scene.normals is not None:
        normal_lengths = np.linalg.norm(scene.normals, axis=-1)
        norm_values = normal_lengths[valid_depth]
        normals_report = {
            "available": True,
            "mean_length_on_valid_depth": _percentile(norm_values, 50),
            "p95_unit_length_error": _percentile(np.abs(norm_values - 1.0), 95),
            "degenerate_fraction_on_valid_depth": float(np.mean(norm_values < 1e-5)) if norm_values.size else 1.0,
        }
    else:
        normals_report = {"available": False}

    semantic_stats = {key: _mask_stats(mask) for key, mask in scene.semantic_masks.items()}
    instance_stats = {key: _mask_stats(mask) for key, mask in scene.instance_masks.items()}

    duplicate_instances: list[dict[str, Any]] = []
    for (name_a, mask_a), (name_b, mask_b) in combinations(scene.instance_masks.items(), 2):
        a = np.asarray(mask_a) > 0.5
        b = np.asarray(mask_b) > 0.5
        union = np.count_nonzero(a | b)
        iou = float(np.count_nonzero(a & b) / union) if union else 0.0
        if iou >= 0.98:
            duplicate_instances.append({"instance_a": name_a, "instance_b": name_b, "iou": iou})

    warnings: list[str] = []
    errors: list[str] = []
    visible_pixel_count = max(int(np.count_nonzero(visible)), 1)
    valid_depth_fraction = float(np.count_nonzero(valid_depth) / visible_pixel_count)
    if valid_depth_fraction < 0.80:
        warnings.append("Less than 80% of visible pixels have valid depth.")
    if depth_spread is None or depth_spread < 1e-4:
        warnings.append("Depth map is flat or has insufficient dynamic range.")
    if depth_bin_count < 8:
        warnings.append("Depth map has fewer than 8 occupied quantile bins; check quantization or model collapse.")
    if confidence.size == 0 or float(np.mean(confidence)) < 0.20:
        warnings.append("Mean depth confidence is low or unavailable over visible pixels.")
    if not normals_report["available"]:
        warnings.append("Surface normals are absent; normal-dependent lighting must not be treated as validated.")
    elif normals_report["degenerate_fraction_on_valid_depth"] > 0.01:
        warnings.append("More than 1% of valid-depth pixels have degenerate surface normals.")
    if geometry_conf is None:
        warnings.append("Geometry confidence map is absent.")
    elif float(np.mean(geometry_conf)) < 0.20:
        warnings.append("Mean geometry confidence is low.")
    if lightable.size == 0 or float(np.mean(lightable > 0.05)) < 1e-4:
        warnings.append("No visible pixels are currently eligible for relighting; review masks and alpha/protection policy.")
    if len(scene.semantic_masks) == 0:
        warnings.append("No semantic class masks were supplied; all visible pixels remain candidates by neutral policy.")
    for name, stats in semantic_stats.items():
        coverage = stats["binary_coverage_fraction"]
        if coverage < 1e-5:
            warnings.append(f"Semantic mask '{name}' is empty.")
        elif coverage > 0.995:
            warnings.append(f"Semantic mask '{name}' covers almost the full image; review detector/mask output.")
    for name, stats in instance_stats.items():
        coverage = stats["binary_coverage_fraction"]
        if coverage < 1e-5:
            warnings.append(f"Instance mask '{name}' is empty.")
        elif coverage > 0.995:
            warnings.append(f"Instance mask '{name}' covers almost the full image; review detector/mask output.")
    if duplicate_instances:
        warnings.append("Two or more instance masks are near-identical; inspect duplicate detections.")

    if not np.isfinite(scene.rgb_srgb).all():
        errors.append("RGB contains non-finite values.")
    status = "structurally_valid_with_warnings" if warnings else "structurally_valid"
    if errors:
        status = "invalid"

    return {
        "schema_version": "1.0",
        "status": status,
        "assessment_scope": "structural_and_degeneracy_checks_only_not_ground_truth_quality",
        "image": {"width": int(w), "height": int(h), "pixels": int(h * w),
                  "depth_kind": scene.depth_kind,
                  "visible_fraction": float(np.mean(visible)),
                  "alpha_present": scene.alpha_mask is not None},
        "depth": {
            "valid_fraction_of_visible_pixels": valid_depth_fraction,
            "p02": depth_p02, "p50": depth_p50, "p98": depth_p98,
            "relative_or_metric_spread": depth_spread,
            "occupied_quantile_bins_64": depth_bin_count,
            "log_gradient_p95": depth_grad_p95,
            "confidence_mean": float(np.mean(confidence)) if confidence.size else None,
            "confidence_below_0_2_fraction": float(np.mean(confidence < 0.2)) if confidence.size else None,
        },
        "geometry": {
            "normals": normals_report,
            "geometry_confidence_mean": float(np.mean(geometry_conf)) if geometry_conf is not None and geometry_conf.size else None,
            "geometry_confidence_below_0_2_fraction": float(np.mean(geometry_conf < 0.2)) if geometry_conf is not None and geometry_conf.size else None,
        },
        "masks": {
            "lightable_fraction_above_0_5": float(np.mean(lightable > 0.5)) if lightable.size else None,
            "protected_fraction_above_0_5": float(np.mean(protected > 0.5)) if protected is not None and protected.size else None,
            "semantic_class_count": len(semantic_stats),
            "instance_count": len(instance_stats),
            "semantic": semantic_stats,
            "instances": instance_stats,
            "near_duplicate_instances": duplicate_instances,
        },
        "materials": {
            "roughness_available": scene.roughness is not None,
            "metalness_available": scene.metalness is not None,
            "specular_level_available": scene.specular_level is not None,
            "material_estimation_claimed": bool(scene.metadata.get("material_maps_estimated", False)),
        },
        "warnings": warnings,
        "errors": errors,
        "metadata": scene.metadata,
        "non_claims": [
            "A structurally valid depth map may still be geometrically wrong.",
            "No ground-truth depth, mask, normal, material, or lighting comparison is performed.",
            "Relative depth is not metric distance.",
            "A detector confidence or mask score is not a calibrated probability unless separately calibrated.",
        ],
    }
