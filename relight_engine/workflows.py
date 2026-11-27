from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .contracts import (
    SCENE_SCHEMA_VERSION,
    CameraModel,
    EngineSettings,
    EnvironmentLight,
    LightingRig,
    Light,
    SceneBuffers,
    validate_scene_schema_version,
)
from .color import srgb_to_linear
from .calibration import (
    MaterialAssignment,
    SourceIllumination,
    SourceLightCalibration,
    build_material_maps,
    kelvin_to_linear_rgb,
)
from .engine import relight
from .image_io import save_image
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


def _light_from_dict(item: dict, index: int, *, prefix: str = "target_light") -> Light:
    if not isinstance(item, dict):
        raise ValueError(f"lights[{index}] must be an object.")
    allowed = {
        "name", "light_id", "kind", "strength", "color", "color_temperature_kelvin", "direction",
        "position", "area_size", "samples", "casts_shadows", "wrap", "radius",
        "spot_inner_angle_deg", "spot_outer_angle_deg", "intensity_mode",
    }
    unknown = set(item) - allowed
    if unknown:
        raise ValueError(f"Unknown light fields at index {index}: {sorted(unknown)}")
    if "color_temperature_kelvin" in item and "color" in item:
        raise ValueError("Specify either color_temperature_kelvin or linear-RGB color, not both.")
    color = (kelvin_to_linear_rgb(float(item["color_temperature_kelvin"]))
             if "color_temperature_kelvin" in item else tuple(item.get("color", (1.0, 1.0, 1.0))))
    return Light(
        name=str(item.get("name", f"{prefix}_{index + 1}")),
        light_id=item.get("light_id"),
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
        spot_inner_angle_deg=float(item.get("spot_inner_angle_deg", 20.0)),
        spot_outer_angle_deg=float(item.get("spot_outer_angle_deg", 40.0)),
        intensity_mode=str(item.get("intensity_mode", "artistic")),  # type: ignore[arg-type]
    )


def source_illumination_from_dict(data: dict) -> SourceIllumination:
    """Parse general SourceIllumination or legacy single-key SourceLightCalibration dict."""
    if not isinstance(data, dict):
        raise ValueError("source_illumination / source_calibration must be an object.")
    if "key_direction" in data and "lights" not in data:
        return SourceIllumination.from_single_key(source_calibration_from_dict(data))
    allowed = {
        "lights", "ambient_strength", "ambient_color", "ambient_temperature_kelvin",
        "ambient_ground_color", "environment", "sh_coefficients", "exposure_scale",
        "white_balance", "confidence", "provenance",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown source_illumination fields: {sorted(unknown)}")
    if "ambient_temperature_kelvin" in data and "ambient_color" in data:
        raise ValueError("Specify either ambient_temperature_kelvin or ambient_color, not both.")
    amb_col = (
        kelvin_to_linear_rgb(float(data["ambient_temperature_kelvin"]))
        if "ambient_temperature_kelvin" in data
        else tuple(float(x) for x in data.get("ambient_color", (1.0, 1.0, 1.0)))
    )
    lights = tuple(_light_from_dict(item, i, prefix="source_light") for i, item in enumerate(data.get("lights", [])))
    sh_coeffs = None if data.get("sh_coefficients") is None else np.asarray(data["sh_coefficients"], dtype=np.float32)
    env_obj = None
    if data.get("environment") is not None:
        env_obj = target_rig_from_dict({"environment": data["environment"]}).environment
    illum = SourceIllumination(
        lights=lights,
        ambient_color=amb_col,  # type: ignore[arg-type]
        ambient_strength=float(data.get("ambient_strength", 0.15)),
        ambient_ground_color=None if data.get("ambient_ground_color") is None else tuple(float(x) for x in data["ambient_ground_color"]),  # type: ignore[arg-type]
        environment=env_obj,
        sh_coefficients=sh_coeffs,
        exposure_scale=float(data.get("exposure_scale", 1.0)),
        white_balance=tuple(float(x) for x in data.get("white_balance", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
        confidence=float(data.get("confidence", 1.0)),
        provenance=dict(data.get("provenance", {"source": "user_manual_source_illumination", "reviewed": True})),
    )
    illum.validate()
    return illum


def target_rig_from_dict(data: dict) -> LightingRig:
    if not isinstance(data, dict):
        raise ValueError("target_rig must be an object.")
    allowed = {
        "ambient_strength", "ambient_temperature_kelvin", "ambient_color",
        "ambient_ground_color", "ambient_ground_temperature_kelvin", "lights", "environment",
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
    lights = [_light_from_dict(item, index, prefix="target_light") for index, item in enumerate(data.get("lights", []))]
    env_obj = None
    if data.get("environment") is not None:
        env_raw = data["environment"]
        if not isinstance(env_raw, dict):
            raise ValueError("target_rig.environment must be an object.")
        env_allowed = {
            "sky_color", "horizon_color", "ground_color", "strength",
            "key_azimuth_deg", "key_elevation_deg", "key_color", "key_strength", "key_sharpness",
            "env_map", "env_map_path", "env_map_strength", "diffuse_strength", "specular_strength", "rotation_deg",
        }
        env_unknown = set(env_raw) - env_allowed
        if env_unknown:
            raise ValueError(f"Unknown target_rig.environment fields: {sorted(env_unknown)}")
        env_kwargs: dict = {}
        for k, v in env_raw.items():
            if k == "env_map":
                env_kwargs["env_map"] = None if v is None else np.asarray(v, dtype=np.float32)
            elif k == "env_map_path":
                if v is not None:
                    p = Path(str(v))
                    if p.suffix.lower() == ".npy":
                        env_kwargs["env_map"] = np.asarray(np.load(p), dtype=np.float32)
                    else:
                        import cv2
                        raw_env = cv2.imread(str(p), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        if raw_env is None:
                            raise ValueError(f"Failed to load environment map: {p}")
                        rgb_env = cv2.cvtColor(raw_env, cv2.COLOR_BGR2RGB).astype(np.float32)
                        if raw_env.dtype == np.uint8:
                            rgb_env = srgb_to_linear(np.clip(rgb_env / 255.0, 0.0, 1.0))
                        elif raw_env.dtype == np.uint16:
                            rgb_env = srgb_to_linear(np.clip(rgb_env / 65535.0, 0.0, 1.0))
                        env_kwargs["env_map"] = np.maximum(rgb_env, 0.0).astype(np.float32)
            elif k.endswith("_color"):
                env_kwargs[k] = tuple(float(x) for x in v)
            else:
                env_kwargs[k] = float(v)
        env_obj = EnvironmentLight(**env_kwargs)
    rig = LightingRig(
        ambient_color=ambient_color,
        ambient_strength=ambient_strength,
        lights=tuple(lights),
        ambient_ground_color=ambient_ground_color,
        environment=env_obj,
    )
    rig.validate()
    return rig


def serialize_scene_artifacts(
    scene_dir: str | Path,
    scene: SceneBuffers,
    camera: CameraModel,
    manifest: dict | None = None,
) -> tuple[Path, Path]:
    """Canonical serializer persisting all SceneBuffers fields, provenance, and versioned schema."""
    scene.validate()
    camera.validate()
    out_dir = Path(scene_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buffers: dict[str, np.ndarray] = {
        "rgb_srgb": np.asarray(scene.rgb_srgb, dtype=np.float32),
        "depth": np.asarray(scene.depth, dtype=np.float32),
        "depth_confidence": np.asarray(scene.depth_confidence, dtype=np.float32),
        "lightable_mask": np.asarray(scene.lightable_mask, dtype=np.float32),
    }
    optional_fields = (
        "normals",
        "geometry_confidence",
        "alpha_mask",
        "protected_mask",
        "roughness",
        "metalness",
        "specular_level",
        "specular_color",
        "clearcoat",
        "clearcoat_roughness",
        "subsurface",
        "anisotropy",
        "anisotropy_angle",
        "material_confidence",
        "albedo",
        "source_shading",
        "non_diffuse_residual",
    )
    for field_name in optional_fields:
        val = getattr(scene, field_name)
        if val is not None:
            buffers[field_name] = np.asarray(val, dtype=np.float32)

    for raw_name, raw_arr in scene.raw_material_maps.items():
        buffers[f"raw_material_{raw_name}"] = np.asarray(raw_arr, dtype=np.float32)
    for label, mask in scene.semantic_masks.items():
        buffers[f"mask_{label}"] = np.asarray(mask, dtype=np.float32)
    for instance_id, mask in scene.instance_masks.items():
        buffers[f"mask_instance_{instance_id}"] = np.asarray(mask, dtype=np.float32)

    buffers_path = out_dir / "scene_buffers.npz"
    np.savez_compressed(buffers_path, **buffers)

    base_manifest = dict(manifest) if manifest is not None else {}
    depth_meta = dict(base_manifest.get("depth", scene.metadata.get("depth", {})))
    depth_meta["depth_kind"] = scene.depth_kind
    base_manifest["scene_schema_version"] = SCENE_SCHEMA_VERSION
    base_manifest["depth"] = depth_meta
    base_manifest["camera"] = {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        **(
            {
                k: v
                for k, v in dict(base_manifest.get("camera", {})).items()
                if k not in ("fx", "fy", "cx", "cy")
            }
        ),
    }
    base_manifest["material_metadata"] = dict(scene.material_metadata)
    base_manifest["intrinsic_metadata"] = dict(scene.intrinsic_metadata)
    base_manifest["scene_metadata"] = dict(scene.metadata)
    base_manifest["semantic_labels"] = sorted(scene.semantic_masks.keys())
    base_manifest["instance_masks"] = sorted(scene.instance_masks.keys())
    base_manifest["raw_material_keys"] = sorted(scene.raw_material_maps.keys())

    manifest_path = out_dir / "scene_manifest.json"
    manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
    return manifest_path, buffers_path


def load_scene_artifacts(scene_dir: str | Path) -> tuple[SceneBuffers, CameraModel, dict, dict[str, np.ndarray]]:
    scene_dir = Path(scene_dir)
    manifest_path = scene_dir / "scene_manifest.json"
    buffers_path = scene_dir / "scene_buffers.npz"
    if not manifest_path.is_file() or not buffers_path.is_file():
        raise FileNotFoundError("scene_dir must contain scene_manifest.json and scene_buffers.npz from run_scene_analysis.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = validate_scene_schema_version(
        manifest.get("scene_schema_version", manifest.get("schema_version"))
    )
    with np.load(buffers_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    required = {"rgb_srgb", "depth", "depth_confidence", "lightable_mask"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"scene_buffers.npz missing arrays: {sorted(missing)}")
    depth_kind = manifest.get("depth", {}).get("depth_kind")
    if depth_kind not in ("metric", "relative"):
        raise ValueError("Scene manifest must declare depth.depth_kind as metric or relative.")

    raw_material_maps = {
        key[len("raw_material_"):]: value
        for key, value in arrays.items()
        if key.startswith("raw_material_")
    }
    saved_meta = dict(manifest.get("scene_metadata", {}))
    merged_meta = {
        **saved_meta,
        "source_manifest": str(manifest_path),
        "scene_schema_version": schema_version,
        "depth": manifest.get("depth", saved_meta.get("depth", {})),
        "segmentation": manifest.get("segmentation", saved_meta.get("segmentation", {})),
        "scene_discovery": manifest.get("scene_discovery", saved_meta.get("scene_discovery", {})),
        "geometry_confidence_source": manifest.get("geometry_confidence", {}).get(
            "source", saved_meta.get("geometry_confidence_source")
        ),
        "neural_scene_estimation_used": bool(
            saved_meta.get("neural_scene_estimation_used", False)
            or manifest.get("depth", {}).get("model_id")
            or manifest.get("segmentation", {}).get("detector_id")
            or manifest.get("scene_discovery", {}).get("model_id")
        ),
    }

    scene = SceneBuffers(
        rgb_srgb=arrays["rgb_srgb"],
        depth=arrays["depth"],
        depth_confidence=arrays["depth_confidence"],
        lightable_mask=arrays["lightable_mask"],
        depth_kind=depth_kind,
        normals=arrays.get("normals"),
        geometry_confidence=arrays.get("geometry_confidence"),
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
        clearcoat=arrays.get("clearcoat"),
        clearcoat_roughness=arrays.get("clearcoat_roughness"),
        subsurface=arrays.get("subsurface"),
        anisotropy=arrays.get("anisotropy"),
        anisotropy_angle=arrays.get("anisotropy_angle"),
        material_confidence=arrays.get("material_confidence"),
        material_metadata=manifest.get("material_metadata", {}),
        albedo=arrays.get("albedo"),
        source_shading=arrays.get("source_shading"),
        non_diffuse_residual=arrays.get("non_diffuse_residual"),
        intrinsic_metadata=manifest.get("intrinsic_metadata", {}),
        raw_material_maps=raw_material_maps,
        schema_version=schema_version,
        metadata=merged_meta,
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
            "anisotropy", "anisotropy_angle", "blend", "confidence",
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
            anisotropy=float(item.get("anisotropy", 0.0)),
            anisotropy_angle=float(item.get("anisotropy_angle", 0.0)),
            confidence=float(item.get("confidence", 1.0)),
        ))
    had_model_priors = bool(
        scene.metadata.get("material_maps_estimated", False)
        or (scene.roughness is not None and any(
            isinstance(v, dict) and v.get("source") == "model_estimated"
            for v in (scene.material_metadata or {}).values()
        ))
    )
    prior_roughness = scene.roughness.copy() if scene.roughness is not None else None
    prior_metalness = scene.metalness.copy() if scene.metalness is not None else None
    prior_spec_level = scene.specular_level.copy() if scene.specular_level is not None else None
    prior_spec_color = scene.specular_color.copy() if scene.specular_color is not None else None
    prior_conf = scene.material_confidence.copy() if scene.material_confidence is not None else None

    maps = build_material_maps(scene.shape, assignments)
    if assignments:
        cov = maps.coverage
        inv = 1.0 - cov
        def _overlay_prior(prior: np.ndarray | None, user_arr: np.ndarray, base_val) -> np.ndarray:
            if prior is None:
                return user_arr
            user_delta = user_arr - inv[..., None] * np.asarray(base_val, dtype=np.float32) if user_arr.ndim == 3 else (user_arr - inv * float(base_val))
            prior_weight = inv[..., None] if user_arr.ndim == 3 else inv
            return np.clip(prior * prior_weight + user_delta, 0.0, 1.0).astype(np.float32)

        scene.roughness = _overlay_prior(prior_roughness, maps.roughness, 0.75)
        scene.metalness = _overlay_prior(prior_metalness, maps.metalness, 0.0)
        scene.specular_level = _overlay_prior(prior_spec_level, maps.specular_level, 0.04)
        if prior_conf is not None:
            scene.material_confidence = np.clip(prior_conf * inv + maps.confidence * cov, 0.0, 1.0).astype(np.float32)
        else:
            scene.material_confidence = maps.confidence
        if maps.specular_color is not None:
            scene.specular_color = _overlay_prior(prior_spec_color, maps.specular_color, (0.65, 0.65, 0.65))
        if maps.clearcoat is not None:
            scene.clearcoat = maps.clearcoat
        if maps.clearcoat_roughness is not None:
            scene.clearcoat_roughness = maps.clearcoat_roughness
        if maps.subsurface is not None:
            scene.subsurface = maps.subsurface
        if maps.anisotropy is not None:
            scene.anisotropy = maps.anisotropy
        if maps.anisotropy_angle is not None:
            scene.anisotropy_angle = maps.anisotropy_angle

        map_source = "user_override_over_model_priors" if had_model_priors else "user_assignment_plus_renderer_prior"
        scalar_map_metadata = {
            "source": map_source,
            "calibrated": False,
            "notes": (
                "Per-mask user overrides blended over model-estimated material priors."
                if had_model_priors
                else "Per-mask user overrides blended over conservative renderer priors; unassigned pixels retain the prior."
            ),
        }
        existing_meta = dict(scene.material_metadata) if isinstance(scene.material_metadata, dict) else {}
        existing_meta.update({
            "roughness": dict(scalar_map_metadata),
            "metalness": dict(scalar_map_metadata),
            "specular_level": dict(scalar_map_metadata),
            "material_confidence": {
                "source": "user_override_plus_model_confidence" if had_model_priors else "user_declared_assignment_confidence",
                "calibrated": False,
                "notes": (
                    "User assignment confidence in masked areas merged with model estimation confidence elsewhere."
                    if had_model_priors
                    else "Caller-supplied reliability weight; zero outside assigned mask coverage. Not a calibrated probability."
                ),
            },
        })
        if maps.specular_color is not None:
            existing_meta["specular_color"] = {
                "source": map_source,
                "calibrated": False,
                "notes": "Linear-RGB conductor F0 user overrides; unassigned pixels retain existing priors.",
            }
        existing_meta["confidence_applied"] = bool(had_model_priors)
        scene.material_metadata = existing_meta
        scene.metadata["material_maps_estimated"] = had_model_priors
    meta_out = dict(maps.metadata)
    meta_out["neural_material_estimation_used"] = had_model_priors
    return meta_out


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
    if config.get("source_illumination") is not None:
        source_illum = source_illumination_from_dict(config["source_illumination"])
        source_rig = source_illum.to_rig()
        source_meta = source_illum.to_metadata()
    else:
        source_calibration = source_calibration_from_dict(config.get("source_calibration"))
        source_rig = source_calibration.to_rig()
        source_meta = source_calibration.to_metadata()
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

    out = save_image(output_path, result.rgb_srgb, alpha_mask=scene.alpha_mask)
    relative_depth_point_area = (
        scene.depth_kind == "relative"
        and any(light.kind in ("point", "area") for light in target_rig.lights)
    )
    report = {
        "status": "rendered_with_explicit_user_calibration",
        "output": str(out.resolve()),
        "source_calibration": source_meta,
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
