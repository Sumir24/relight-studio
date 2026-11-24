"""Structural QA and conservative rendering-readiness diagnostics.

These checks detect malformed or degenerate scene buffers. They do not establish
semantic correctness, geometric ground truth, calibrated uncertainty, or perceptual
relighting quality; those require annotated real-photo benchmarks and human review.
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


def _confidence_kind(source: Any) -> str:
    """Classify provenance text without assuming that a score is calibrated."""
    if source is None or not str(source).strip():
        return "unspecified"
    text = str(source).casefold()
    if "constant" in text and "prior" in text:
        return "constant_prior"
    if "not calibrated" in text or "uncalibrated" in text:
        return "heuristic" if "heuristic" in text else "uncalibrated_score"
    if "calibrated" in text:
        return "calibrated"
    if "heuristic" in text:
        return "heuristic"
    if "ranking" in text or "score" in text:
        return "uncalibrated_score"
    return "declared_source_unclassified"


def describe_material_response(scene: SceneBuffers) -> dict[str, Any]:
    """Describe supplied material maps, provenance and the renderer's explicit fallbacks."""
    metadata = scene.material_metadata if isinstance(scene.material_metadata, dict) else {}
    map_specs = {
        "roughness": {
            "array": scene.roughness,
            "prior": 0.75,
            "shape": "HxW",
        },
        "metalness": {
            "array": scene.metalness,
            "prior": 0.0,
            "shape": "HxW",
        },
        "specular_level": {
            "array": scene.specular_level,
            "prior": 0.04,
            "shape": "HxW",
        },
        "specular_color": {
            "array": scene.specular_color,
            "prior": [0.65, 0.65, 0.65],
            "shape": "HxWx3_linear_RGB_conductor_F0",
        },
    }

    maps: dict[str, dict[str, Any]] = {}
    for name, spec in map_specs.items():
        supplied = spec["array"]
        entry = metadata.get(name, {})
        if isinstance(entry, str):
            source = entry
            calibrated = False
            notes = None
        elif isinstance(entry, dict):
            source = str(entry.get("source", "unspecified"))
            calibrated = bool(entry.get("calibrated", False))
            notes = entry.get("notes")
        else:
            source = "unspecified"
            calibrated = False
            notes = None

        values = None if supplied is None else np.asarray(supplied, dtype=np.float32)
        maps[name] = {
            "available": supplied is not None,
            "source": source if supplied is not None else "conservative_prior",
            "calibrated": calibrated,
            "expected_shape": spec["shape"],
            "fallback_prior": spec["prior"],
            "mean": (
                [float(v) for v in np.mean(values, axis=(0, 1))]
                if values is not None and values.ndim == 3
                else (float(np.mean(values)) if values is not None else None)
            ),
            "minimum": float(np.min(values)) if values is not None else None,
            "maximum": float(np.max(values)) if values is not None else None,
            "notes": notes,
        }

    confidence = scene.material_confidence
    confidence_values = (
        np.asarray(confidence, dtype=np.float32) if confidence is not None else None
    )
    confidence_entry = metadata.get("material_confidence", {})
    if isinstance(confidence_entry, dict):
        confidence_source = str(confidence_entry.get("source", "unspecified"))
        confidence_calibrated = bool(confidence_entry.get("calibrated", False))
    elif isinstance(confidence_entry, str):
        confidence_source = confidence_entry
        confidence_calibrated = False
    else:
        confidence_source = "unspecified"
        confidence_calibrated = False

    unimplemented = [
        "layered_clear_coat",
        "subsurface_scattering",
        "transmission_and_refraction",
        "emissive_radiance_separation",
        "environment_dependent_reflection",
    ]
    return {
        "maps": maps,
        "material_confidence": {
            "available": confidence_values is not None,
            "source": confidence_source if confidence_values is not None else "unavailable",
            "calibrated": confidence_calibrated,
            "mean": float(np.mean(confidence_values)) if confidence_values is not None else None,
            "minimum": float(np.min(confidence_values)) if confidence_values is not None else None,
            "maximum": float(np.max(confidence_values)) if confidence_values is not None else None,
            "semantics": (
                "Blends supplied material maps toward declared conservative priors."
                if confidence_values is not None
                else "No per-pixel confidence blend; supplied maps are used as provided."
            ),
        },
        "implemented_response_families": [
            "lambertian_diffuse_baseline",
            "ggx_rough_dielectric",
            "ggx_conductor_with_RGB_F0",
        ],
        "not_implemented_response_families": unimplemented,
        "material_estimation_claimed": bool(
            scene.metadata.get("material_maps_estimated", False)
        ),
        "interpretation": (
            "Material maps are inputs, not inferred ground truth. Unavailable maps use the declared priors."
        ),
    }


def _confidence_provenance(scene: SceneBuffers) -> dict[str, dict[str, Any]]:
    metadata = scene.metadata if isinstance(scene.metadata, dict) else {}
    depth_metadata = metadata.get("depth", {})
    if not isinstance(depth_metadata, dict):
        depth_metadata = {}

    depth_source = depth_metadata.get(
        "confidence_source", metadata.get("depth_confidence_source")
    )
    geometry_source = metadata.get("geometry_confidence_source")
    depth_kind = _confidence_kind(depth_source)
    geometry_kind = _confidence_kind(geometry_source)
    return {
        "depth": {
            "source": str(depth_source) if depth_source is not None else "unspecified",
            "kind": depth_kind,
            "calibrated": depth_kind == "calibrated",
        },
        "geometry": {
            "source": str(geometry_source) if geometry_source is not None else "unspecified",
            "kind": geometry_kind,
            "calibrated": geometry_kind == "calibrated",
        },
    }


def evaluate_scene_buffers(scene: SceneBuffers) -> dict[str, Any]:
    """Validate buffers and report structural readiness, not perceptual accuracy."""
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
        depth_spread = float(
            (depth_p98 - depth_p02) / max(abs(depth_p50), 1e-6)
        )

    # Quantile bins avoid a large unique() allocation on high-resolution float maps.
    depth_bin_count = 0
    if depth_values.size:
        lo, hi = np.percentile(depth_values, [1, 99])
        if hi > lo:
            bins = np.clip(
                np.floor((depth_values - lo) / (hi - lo) * 63), 0, 63
            ).astype(np.uint8)
            depth_bin_count = int(np.unique(bins).size)

    depth_grad_p95 = None
    if np.count_nonzero(valid_depth) >= 16:
        dx_valid = valid_depth[:, 1:] & valid_depth[:, :-1]
        dy_valid = valid_depth[1:, :] & valid_depth[:-1, :]
        dx = np.abs(
            np.diff(np.log(np.maximum(depth, 1e-6)), axis=1)
        )[dx_valid]
        dy = np.abs(
            np.diff(np.log(np.maximum(depth, 1e-6)), axis=0)
        )[dy_valid]
        grads = (
            np.concatenate([dx, dy])
            if dx.size or dy.size
            else np.array([], dtype=np.float32)
        )
        depth_grad_p95 = _percentile(grads, 95)

    confidence = scene.depth_confidence[visible]
    geometry_conf = (
        None
        if scene.geometry_confidence is None
        else scene.geometry_confidence[visible]
    )
    lightable = scene.lightable_mask[visible]
    protected = (
        None if scene.protected_mask is None else scene.protected_mask[visible]
    )
    provenance = _confidence_provenance(scene)

    normals_report: dict[str, Any]
    if scene.normals is not None:
        normal_lengths = np.linalg.norm(scene.normals, axis=-1)
        norm_values = normal_lengths[valid_depth]
        normals_report = {
            "available": True,
            "median_length_on_valid_depth": _percentile(norm_values, 50),
            "p95_unit_length_error": _percentile(
                np.abs(norm_values - 1.0), 95
            ),
            "degenerate_fraction_on_valid_depth": (
                float(np.mean(norm_values < 1e-5)) if norm_values.size else 1.0
            ),
        }
    else:
        normals_report = {
            "available": False,
            "median_length_on_valid_depth": None,
            "p95_unit_length_error": None,
            "degenerate_fraction_on_valid_depth": 1.0,
        }

    semantic_stats = {
        key: _mask_stats(mask) for key, mask in scene.semantic_masks.items()
    }
    instance_stats = {
        key: _mask_stats(mask) for key, mask in scene.instance_masks.items()
    }

    duplicate_instances: list[dict[str, Any]] = []
    for (name_a, mask_a), (name_b, mask_b) in combinations(
        scene.instance_masks.items(), 2
    ):
        a = np.asarray(mask_a) > 0.5
        b = np.asarray(mask_b) > 0.5
        union = np.count_nonzero(a | b)
        iou = float(np.count_nonzero(a & b) / union) if union else 0.0
        if iou >= 0.98:
            duplicate_instances.append(
                {"instance_a": name_a, "instance_b": name_b, "iou": iou}
            )

    warnings: list[str] = []
    errors: list[str] = []
    visible_pixel_count = max(int(np.count_nonzero(visible)), 1)
    valid_depth_fraction = float(
        np.count_nonzero(valid_depth) / visible_pixel_count
    )
    depth_confidence_mean = (
        float(np.mean(confidence)) if confidence.size else None
    )
    geometry_confidence_mean = (
        float(np.mean(geometry_conf))
        if geometry_conf is not None and geometry_conf.size
        else None
    )

    if valid_depth_fraction < 0.80:
        warnings.append("Less than 80% of visible pixels have valid depth.")
    if depth_spread is None or depth_spread < 1e-4:
        warnings.append(
            "Depth map is flat or has insufficient dynamic range; this can be valid for a fronto-parallel plane but may also indicate model collapse."
        )
    if depth_bin_count < 8:
        warnings.append(
            "Depth map has fewer than 8 occupied quantile bins; inspect the scene and checkpoint before relying on depth variation."
        )
    if depth_confidence_mean is None or depth_confidence_mean < 0.20:
        warnings.append("Mean depth confidence is low or unavailable over visible pixels.")
    if provenance["depth"]["kind"] == "constant_prior":
        warnings.append(
            "Depth confidence is a constant prior, not calibrated per-pixel uncertainty."
        )
    if not normals_report["available"]:
        warnings.append(
            "Surface normals are absent from the saved scene buffers; the renderer may estimate them from depth, but the saved normal map is not available for inspection."
        )
    elif normals_report["degenerate_fraction_on_valid_depth"] > 0.01:
        warnings.append(
            "More than 1% of valid-depth pixels have degenerate surface normals."
        )
    if geometry_conf is None:
        warnings.append("Geometry confidence map is absent.")
    elif geometry_confidence_mean is not None and geometry_confidence_mean < 0.20:
        warnings.append("Mean geometry confidence is low.")
    if lightable.size == 0 or float(np.mean(lightable > 0.05)) < 1e-4:
        warnings.append(
            "No visible pixels are currently eligible for relighting; review masks and alpha/protection policy."
        )
    if len(scene.semantic_masks) == 0:
        warnings.append(
            "No semantic class masks were supplied; all visible pixels remain candidates by neutral policy."
        )
    for name, stats in semantic_stats.items():
        coverage = stats["binary_coverage_fraction"]
        if coverage < 1e-5:
            warnings.append(f"Semantic mask '{name}' is empty.")
        elif coverage > 0.995:
            warnings.append(
                f"Semantic mask '{name}' covers almost the full image; review detector/mask output."
            )
    for name, stats in instance_stats.items():
        coverage = stats["binary_coverage_fraction"]
        if coverage < 1e-5:
            warnings.append(f"Instance mask '{name}' is empty.")
        elif coverage > 0.995:
            warnings.append(
                f"Instance mask '{name}' covers almost the full image; review detector/mask output."
            )
    if duplicate_instances:
        warnings.append(
            "Two or more instance masks are near-identical; inspect duplicate detections."
        )

    material_report = describe_material_response(scene)
    supplied_material_maps = [
        name for name, detail in material_report["maps"].items()
        if detail["available"]
    ]
    maps_without_source = [
        name for name in supplied_material_maps
        if material_report["maps"][name]["source"] == "unspecified"
    ]
    if maps_without_source:
        warnings.append(
            "Material maps lack declared provenance: " + ", ".join(maps_without_source) + "."
        )
    model_estimated_maps = [
        name for name in supplied_material_maps
        if material_report["maps"][name]["source"].casefold() in {
            "model_estimated", "model-estimated", "estimated"
        }
    ]
    if model_estimated_maps and not material_report["material_confidence"]["available"]:
        warnings.append(
            "Model-estimated material maps have no per-pixel material-confidence map."
        )
    if scene.specular_color is not None and scene.metalness is None:
        warnings.append(
            "RGB conductor F0 is supplied without a metalness map; the conservative zero-metalness prior suppresses its conductor contribution."
        )
    if scene.material_confidence is not None and not supplied_material_maps:
        warnings.append(
            "Material confidence is supplied but no material property maps are available to blend."
        )

    if not np.isfinite(scene.rgb_srgb).all():
        errors.append("RGB contains non-finite values.")

    # This is a structural gate only. It does not certify geometric correctness.
    readiness_reasons: list[str] = []
    if valid_depth_fraction < 0.80:
        readiness_reasons.append("valid depth covers less than 80% of visible pixels")
    if not normals_report["available"]:
        readiness_reasons.append("no stored normal map is available")
    elif normals_report["degenerate_fraction_on_valid_depth"] > 0.01:
        readiness_reasons.append("more than 1% of valid-depth normals are degenerate")
    if geometry_confidence_mean is None:
        readiness_reasons.append("geometry confidence is unavailable")
    elif geometry_confidence_mean < 0.20:
        readiness_reasons.append("mean geometry confidence is below 0.20")
    if depth_confidence_mean is None or depth_confidence_mean < 0.20:
        readiness_reasons.append("mean depth confidence is below 0.20")
    if lightable.size == 0 or not np.any(lightable > 0.05):
        readiness_reasons.append("no visible pixels are eligible for relighting")

    structural_normal_gate = not readiness_reasons
    metric_depth_available = scene.depth_kind == "metric"
    status = (
        "structurally_valid_with_warnings" if warnings else "structurally_valid"
    )
    if errors:
        status = "invalid"

    return {
        "schema_version": "1.1",
        "status": status,
        "assessment_scope": "structural_and_degeneracy_checks_only_not_ground_truth_quality",
        "image": {
            "width": int(w),
            "height": int(h),
            "pixels": int(h * w),
            "depth_kind": scene.depth_kind,
            "visible_fraction": float(np.mean(visible)),
            "alpha_present": scene.alpha_mask is not None,
        },
        "depth": {
            "valid_fraction_of_visible_pixels": valid_depth_fraction,
            "p02": depth_p02,
            "p50": depth_p50,
            "p98": depth_p98,
            "relative_or_metric_spread": depth_spread,
            "occupied_quantile_bins_64": depth_bin_count,
            "log_gradient_p95": depth_grad_p95,
            "confidence_mean": depth_confidence_mean,
            "confidence_below_0_2_fraction": (
                float(np.mean(confidence < 0.2)) if confidence.size else None
            ),
        },
        "geometry": {
            "normals": normals_report,
            "geometry_confidence_mean": geometry_confidence_mean,
            "geometry_confidence_below_0_2_fraction": (
                float(np.mean(geometry_conf < 0.2))
                if geometry_conf is not None and geometry_conf.size
                else None
            ),
        },
        "confidence_provenance": provenance,
        "lighting_readiness": {
            "normal_dependent_shading": {
                "structural_gate_passed": structural_normal_gate,
                "blocking_reasons": readiness_reasons,
                "scope": "structural_readiness_only_not_visual_or_ground_truth_validation",
            },
            "metric_distance_falloff": {
                "metric_depth_available": metric_depth_available,
                "reason": (
                    None
                    if metric_depth_available
                    else "Depth is relative; it cannot provide metre-based light distance or physically calibrated distance falloff."
                ),
            },
        },
        "masks": {
            "lightable_fraction_above_0_5": (
                float(np.mean(lightable > 0.5)) if lightable.size else None
            ),
            "protected_fraction_above_0_5": (
                float(np.mean(protected > 0.5))
                if protected is not None and protected.size
                else None
            ),
            "semantic_class_count": len(semantic_stats),
            "instance_count": len(instance_stats),
            "semantic": semantic_stats,
            "instances": instance_stats,
            "near_duplicate_instances": duplicate_instances,
        },
        "materials": material_report,
        "warnings": warnings,
        "errors": errors,
        "metadata": scene.metadata,
        "non_claims": [
            "A structurally valid depth map may still be geometrically wrong.",
            "A structural lighting-readiness pass does not establish perceptual relighting quality.",
            "No ground-truth depth, mask, normal, material, or lighting comparison is performed.",
            "Relative depth is not metric distance.",
            "A detector confidence, mask score, or heuristic reliability value is not a calibrated probability unless separately calibrated.",
            "A constant confidence prior is not per-pixel model uncertainty.",
        ],
    }
