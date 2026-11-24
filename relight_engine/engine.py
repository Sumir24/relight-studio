from __future__ import annotations

from copy import copy

import numpy as np

from .contracts import (
    CameraModel,
    SceneBuffers,
    LightingRig,
    EngineSettings,
    RelightResult,
)
from .geometry import prepare_geometry
from .lighting import evaluate_rig
from .composite import compose_source_preserving
from .quality import evaluate_scene_buffers, describe_material_response
from .color import srgb_to_linear


def relight(
    scene: SceneBuffers,
    camera: CameraModel,
    source_rig: LightingRig,
    target_rig: LightingRig,
    settings: EngineSettings | None = None,
    *,
    require_geometry_ready: bool = False,
) -> RelightResult:
    """Relight an image using shared 2.5D geometry and source-to-target transfer.

    Requires externally estimated depth and scene masks. Does not fabricate geometry.

    Set require_geometry_ready=True to enforce the structural normal-dependent
    lighting gate before rendering. The gate checks buffer coverage and reliability
    diagnostics; it does not certify that estimated geometry is physically correct.
    """
    settings = settings or EngineSettings()
    settings.validate()
    scene.validate()
    camera.validate()

    points, normals, geometry_conf = prepare_geometry(scene, camera)

    gate_report = None
    if require_geometry_ready:
        # QA the exact geometry prepared for this render, including normals estimated
        # on demand when they were not persisted in SceneBuffers.
        quality_scene = copy(scene)
        quality_scene.normals = normals
        quality_scene.geometry_confidence = geometry_conf
        gate_report = evaluate_scene_buffers(quality_scene)
        gate = gate_report["lighting_readiness"]["normal_dependent_shading"]
        if not gate["structural_gate_passed"]:
            reasons = "; ".join(gate["blocking_reasons"])
            raise ValueError(
                "Scene failed the structural gate for normal-dependent lighting: "
                + reasons
            )

    source_d, source_s, _ = evaluate_rig(
        scene, camera, points, normals, source_rig, settings
    )
    target_d, target_s, visibility = evaluate_rig(
        scene, camera, points, normals, target_rig, settings
    )
    out, ratio = compose_source_preserving(
        scene,
        source_d,
        target_d,
        source_s,
        target_s,
        geometry_conf,
        visibility,
        settings,
    )

    active = (
        (scene.lightable_mask > 0.05)
        & (geometry_conf > settings.min_geometry_confidence)
    )
    delta = np.abs(out - scene.rgb_srgb)
    metrics = {
        "mode": "2.5D source-to-target relighting",
        "depth_kind": scene.depth_kind,
        "material_response": describe_material_response(scene),
        "geometry_valid_fraction": float(
            np.mean(
                (scene.depth > 0)
                & (geometry_conf > settings.min_geometry_confidence)
            )
        ),
        "lightable_fraction": float(np.mean(active)),
        "mean_abs_srgb_change_active": (
            float(np.mean(delta[active])) if np.any(active) else 0.0
        ),
        "mean_abs_srgb_change_protected": (
            float(np.mean(delta[~active])) if np.any(~active) else 0.0
        ),
        "mean_target_to_source_shading_ratio": (
            float(np.mean(ratio[active])) if np.any(active) else 1.0
        ),
        "shadow_visibility_mean": (
            float(np.mean(visibility[active])) if np.any(active) else 1.0
        ),
        "geometry_readiness_gate_enforced": bool(require_geometry_ready),
        "geometry_readiness_gate": (
            gate_report["lighting_readiness"]["normal_dependent_shading"]
            if gate_report is not None
            else None
        ),
        "neural_scene_estimation_used": bool(
            scene.metadata.get("neural_scene_estimation_used", False)
        ),
        "limitations": [
            "2.5D depth represents only the visible first surface; hidden surfaces and overhangs are unavailable.",
            "Relighting quality depends on depth, camera calibration, semantic masks, and source-light estimation.",
            "Shadow visibility is approximate screen-space ray marching, not a full 3D mesh shadow map.",
            "Single-image intrinsic decomposition remains ambiguous; source_rig must approximate the original illumination.",
            "The structural geometry gate is not a substitute for real-photo visual validation.",
            "Relative depth does not support metre-based point/area-light distance falloff.",
        ],
    }
    return RelightResult(
        out, normals, ratio, visibility, geometry_conf, metrics
    )
