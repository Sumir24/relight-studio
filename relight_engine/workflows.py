from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .contracts import CameraModel, EngineSettings, LightingRig, Light, SceneBuffers
from .calibration import (
    MaterialAssignment,
    SourceLightCalibration,
    build_material_maps,
    kelvin_to_linear_rgb,
)
from .engine import relight
from .scene_policy import semantic_key


def source_calibration_from_dict(data: dict) -> SourceLightCalibration:
    if not isinstance(data, dict):
        raise ValueError("source_calibration must be an object.")
    allowed = {
        "key_direction", "key_strength", "key_temperature_kelvin", "ambient_strength",
        "ambient_temperature_kelvin", "casts_shadows", "name",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown source-calibration fields: {sorted(unknown)}")
    if "key_direction" not in data:
        raise ValueError("source_calibration.key_direction is required.")
    calibration = SourceLightCalibration(**data)
    calibration.validate()
    return calibration


def target_rig_from_dict(data: dict) -> LightingRig:
    if not isinstance(data, dict):
        raise ValueError("target_rig must be an object.")
    allowed = {
        "ambient_strength", "ambient_temperature_kelvin", "ambient_color",
        "ambient_ground_color", "ambient_ground_temperature_kelvin", "lights",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown target-rig fields: {sorted(unknown)}")
    ambient_strength = float(data.get("ambient_strength", 0.10))
    if "ambient_temperature_kelvin" in data and "ambient_color" in data:
        raise ValueError("Specify either ambient_temperature_kelvin or ambient_color, not both.")
    if "ambient_temperature_kelvin" in data:
        ambient_color = kelvin_to_linear_rgb(float(data["ambient_temperature_kelvin"]))
    else:
        ambient_color = tuple(data.get("ambient_color", (1.0, 1.0, 1.0)))
    if "ambient_ground_temperature_kelvin" in data and "ambient_ground_color" in data:
        raise ValueError("Specify either ambient_ground_temperature_kelvin or ambient_ground_color, not both.")
    if "ambient_ground_temperature_kelvin" in data:
        ambient_ground_color = kelvin_to_linear_rgb(float(data["ambient_ground_temperature_kelvin"]))
    elif "ambient_ground_color" in data:
        ambient_ground_color = tuple(data["ambient_ground_color"])
    else:
        ambient_ground_color = None
    lights = []
    for index, item in enumerate(data.get("lights", [])):
        if not isinstance(item, dict):
            raise ValueError(f"target_rig.lights[{index}] must be an object.")
        allowed = {
            "name", "kind", "strength", "color", "color_temperature_kelvin", "direction",
            "position", "area_size", "samples", "casts_shadows", "wrap", "radius",
        }
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"Unknown target light fields at index {index}: {sorted(unknown)}")
        if "color_temperature_kelvin" in item and "color" in item:
            raise ValueError("Specify either color_temperature_kelvin or linear-RGB color, not both.")
        color = (kelvin_to_linear_rgb(float(item["color_temperature_kelvin"]))
                 if "color_temperature_kelvin" in item else tuple(item.get("color", (1.0, 1.0, 1.0))))
        lights.append(Light(
            name=str(item.get("name", f"target_light_{index + 1}")),
            kind=item["kind"],
            strength=float(item.get("strength", 1.0)),
            color=color,
            direction=None if item.get("direction") is None else tuple(item["direction"]),
            position=None if item.get("position") is None else tuple(item["position"]),
            area_size=tuple(item.get("area_size", (0.5, 0.5))),
            samples=int(item.get("samples", 4)),
            casts_shadows=bool(item.get("casts_shadows", True)),
            wrap=float(item.get("wrap", 0.0)),
            radius=float(item.get("radius", 0.0)),
        ))
    rig = LightingRig(
        ambient_color=ambient_color,
        ambient_strength=ambient_strength,
        lights=tuple(lights),
        ambient_ground_color=ambient_ground_color,
    )
    rig.validate()
    return rig


def load_scene_artifacts(scene_dir: str | Path) -> tuple[SceneBuffers, CameraModel, dict, dict[str, np.ndarray]]:
    scene_dir = Path(scene_dir)
    manifest_path = scene_dir / "scene_manifest.json"
    buffers_path = scene_dir / "scene_buffers.npz"
    if not manifest_path.is_file() or not buffers_path.is_file():
        raise FileNotFoundError("scene_dir must contain scene_manifest.json and scene_buffers.npz from run_scene_analysis.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(buffers_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    required = {"rgb_srgb", "depth", "depth_confidence", "geometry_confidence", "normals", "lightable_mask"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"scene_buffers.npz missing arrays: {sorted(missing)}")
    depth_kind = manifest.get("depth", {}).get("depth_kind")
    if depth_kind not in ("metric", "relative"):
        raise ValueError("Scene manifest must declare depth.depth_kind as metric or relative.")
    scene = SceneBuffers(
        rgb_srgb=arrays["rgb_srgb"],
        depth=arrays["depth"],
        depth_confidence=arrays["depth_confidence"],
        lightable_mask=arrays["lightable_mask"],
        depth_kind=depth_kind,
        normals=arrays["normals"],
        geometry_confidence=arrays["geometry_confidence"],
        alpha_mask=arrays.get("alpha_mask"),
        protected_mask=arrays.get("protected_mask", arrays.get("mask_protected")),
        semantic_masks={
            key[len("mask_"):]: value for key, value in arrays.items()
            if key.startswith("mask_") and key not in ("mask_lightable", "mask_protected")
            and not key.startswith("mask_instance_")
        },
        instance_masks={
            key[len("mask_instance_"):]: value for key, value in arrays.items()
            if key.startswith("mask_instance_")
        },
        roughness=arrays.get("roughness"),
        metalness=arrays.get("metalness"),
        specular_level=arrays.get("specular_level"),
        specular_color=arrays.get("specular_color"),
        material_confidence=arrays.get("material_confidence"),
        material_metadata=manifest.get("material_metadata", {}),
        albedo=arrays.get("albedo"),
        source_shading=arrays.get("source_shading"),
        non_diffuse_residual=arrays.get("non_diffuse_residual"),
        intrinsic_metadata=manifest.get("intrinsic_metadata", {}),
        metadata={
            "source_manifest": str(manifest_path),
            "depth": manifest.get("depth", {}),
            "segmentation": manifest.get("segmentation", {}),
            "scene_discovery": manifest.get("scene_discovery", {}),
            "geometry_confidence_source": manifest.get("geometry_confidence", {}).get("source"),
            "neural_scene_estimation_used": bool(
                manifest.get("depth", {}).get("model_id")
                or manifest.get("segmentation", {}).get("detector_id")
                or manifest.get("scene_discovery", {}).get("model_id")
            ),
        },
    )
    camera_data = manifest.get("camera", {})
    camera = CameraModel(
        fx=float(camera_data["fx"]), fy=float(camera_data["fy"]),
        cx=float(camera_data["cx"]), cy=float(camera_data["cy"]),
    )
    camera.validate()
    scene.validate()
    return scene, camera, manifest, arrays


def _resolve_mask_array(arrays: dict[str, np.ndarray], mask_key: str) -> np.ndarray:
    """Resolve user-supplied mask references across NPZ keys, PNG filenames, and raw labels."""
    raw = str(mask_key).strip()
    if not raw:
        raise ValueError(f"Material assignment mask {mask_key!r} is not present in scene buffers.")
    stem = raw[:-4] if raw.lower().endswith(".png") else raw
    candidates: list[str] = [raw, stem]
    if not stem.startswith("mask_"):
        candidates.append(f"mask_{stem}")
    if stem.startswith("instance_mask_"):
        candidates.append(f"mask_instance_{stem[len('instance_mask_'):]}")
    try:
        normalized = semantic_key(stem)
    except ValueError:
        normalized = ""
    if normalized:
        candidates.append(f"mask_{normalized}")
        if normalized.startswith("instance_mask_"):
            candidates.append(f"mask_instance_{normalized[len('instance_mask_'):]}")
    for candidate in candidates:
        if candidate in arrays:
            return arrays[candidate]
    raise ValueError(f"Material assignment mask {mask_key!r} is not present in scene buffers.")


def apply_user_material_assignments(
    scene: SceneBuffers,
    arrays: dict[str, np.ndarray],
    config: list[dict],
) -> dict:
    assignments: list[MaterialAssignment] = []
    for index, item in enumerate(config):
        if not isinstance(item, dict):
            raise ValueError(f"material_assignments[{index}] must be an object.")
        allowed = {
            "name", "mask_key", "roughness", "metalness", "specular_level",
            "specular_color", "clearcoat", "clearcoat_roughness", "subsurface",
            "blend", "confidence",
        }
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(f"Unknown material assignment fields at index {index}: {sorted(unknown)}")
        mask_key = str(item.get("mask_key", ""))
        mask = _resolve_mask_array(arrays, mask_key)
        color = item.get("specular_color")
        assignments.append(MaterialAssignment(
            name=str(item["name"]), mask=mask,
            roughness=float(item["roughness"]), metalness=float(item.get("metalness", 0.0)),
            specular_level=float(item.get("specular_level", 0.04)),
            blend=float(item.get("blend", 1.0)),
            specular_color=None if color is None else tuple(float(v) for v in color),
            clearcoat=float(item.get("clearcoat", 0.0)),
            clearcoat_roughness=float(item.get("clearcoat_roughness", 0.15)),
            subsurface=float(item.get("subsurface", 0.0)),
            confidence=float(item.get("confidence", 1.0)),
        ))
    maps = build_material_maps(scene.shape, assignments)
    if assignments:
        scene.roughness = maps.roughness
        scene.metalness = maps.metalness
        scene.specular_level = maps.specular_level
        scene.material_confidence = maps.confidence
        if maps.specular_color is not None:
            scene.specular_color = maps.specular_color
        if maps.clearcoat is not None:
            scene.clearcoat = maps.clearcoat
        if maps.clearcoat_roughness is not None:
            scene.clearcoat_roughness = maps.clearcoat_roughness
        if maps.subsurface is not None:
            scene.subsurface = maps.subsurface
        scalar_map_metadata = {
            "source": "user_assignment_plus_renderer_prior",
            "calibrated": False,
            "notes": "Per-mask user overrides blended over conservative renderer priors; unassigned pixels retain the prior.",
        }
        scene.material_metadata = {
            "roughness": dict(scalar_map_metadata),
            "metalness": dict(scalar_map_metadata),
            "specular_level": dict(scalar_map_metadata),
            "material_confidence": {
                "source": "user_declared_assignment_confidence",
                "calibrated": False,
                "notes": "Caller-supplied reliability weight; zero outside assigned mask coverage. Not a calibrated probability.",
            },
        }
        if maps.specular_color is not None:
            scene.material_metadata["specular_color"] = {
                "source": "user_assignment_plus_renderer_prior",
                "calibrated": False,
                "notes": "Linear-RGB conductor F0 user overrides; unassigned pixels retain the neutral 0.65 RGB prior.",
            }
        scene.metadata["material_maps_estimated"] = False
    return maps.metadata


def run_scene_relight(
    scene_dir: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    *,
    renderer_mode: str | None = None,
    relighter_backend: str | None = None,
    device: str = "auto",
) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("approved_for_test") is not True:
        raise ValueError("Set approved_for_test=true only after reviewing source and target calibration values.")
    scene, camera, manifest, arrays = load_scene_artifacts(scene_dir)
    source_calibration = source_calibration_from_dict(config.get("source_calibration"))
    source_rig = source_calibration.to_rig()
    target_rig = target_rig_from_dict(config.get("target_rig"))
    material_metadata = apply_user_material_assignments(scene, arrays, config.get("material_assignments", []))
    raw_settings = dict(config.get("settings", {}))
    if renderer_mode is not None:
        raw_settings["renderer_mode"] = renderer_mode
    settings = EngineSettings(**raw_settings)
    require_geometry_ready = bool(config.get("require_geometry_ready", True))

    chosen_relighter = relighter_backend if relighter_backend is not None else config.get("relighter_backend")
    relighter = None
    if settings.renderer_mode == "hybrid_neural":
        from .adapters.registry import available_backends, create_relighter_backend

        if not chosen_relighter or str(chosen_relighter).strip().lower() == "none":
            choices = ", ".join(available_backends()["relighters"])
            raise ValueError(
                f"renderer_mode='hybrid_neural' requires a relighter_backend (available: {choices})."
            )
        relighter_kwargs = dict(config.get("relighter_options", {}))
        if device != "auto" or "device" not in relighter_kwargs:
            relighter_kwargs["device"] = device
        relighter = create_relighter_backend(str(chosen_relighter), **relighter_kwargs)

    try:
        result = relight(
            scene, camera, source_rig, target_rig, settings,
            require_geometry_ready=require_geometry_ready,
            neural_relighter=relighter,
        )
    finally:
        if relighter is not None:
            relighter.close()

    from PIL import Image
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb_u8 = np.round(np.clip(result.rgb_srgb, 0, 1) * 255).astype(np.uint8)
    if scene.alpha_mask is not None:
        rgba_u8 = np.dstack([rgb_u8, np.round(np.clip(scene.alpha_mask, 0, 1) * 255).astype(np.uint8)])
        Image.fromarray(rgba_u8, mode="RGBA").save(out)
    else:
        Image.fromarray(rgb_u8, mode="RGB").save(out)
    relative_depth_point_area = (
        scene.depth_kind == "relative"
        and any(light.kind in ("point", "area") for light in target_rig.lights)
    )
    report = {
        "status": "rendered_with_explicit_user_calibration",
        "output": str(out.resolve()),
        "source_calibration": source_calibration.to_metadata(),
        "material_maps": material_metadata,
        "target_rig": config["target_rig"],
        "scene_manifest": str((Path(scene_dir) / "scene_manifest.json").resolve()),
        "engine_metrics": result.metrics,
        "physical_accuracy_notes": [
            "Directional lights are the recommended first target-light type while depth is relative.",
            "Point/area light distance falloff uses normalized scene units when depth is relative; it is not metric inverse-square falloff."
        ] if relative_depth_point_area else [
            "The target rig uses directional lighting or metric depth for point/area placement."
        ],
        "limitations": [
            "Source-light parameters were supplied by the user; they were not inferred automatically.",
            "Material assignments were supplied by the user; generic base material values remain priors.",
            "Depth, camera, segmentation and source-light assumptions must be reviewed before treating the image as physically relit.",
        ],
    }
    Path(str(out) + ".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
