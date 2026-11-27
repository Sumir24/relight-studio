#!/usr/bin/env python3
"""
Relight Studio Core Engine - Direct Execution Script
Solely executes relight_engine.engine.relight (100% deterministic physics rendering, 0% AI).
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import cv2
import numpy as np

from relight_engine.contracts import (
    CameraModel,
    EngineSettings,
    Light,
    LightingRig,
)
from relight_engine.engine import relight
from relight_engine.workflows import load_scene_artifacts

ARTIFACT_DIR = Path("/Users/sumirdayal/.gemini/antigravity/brain/298a0ed6-69cc-40a9-991d-4ea0037c7e9d")

def run_manga_engine():
    scene_dir = Path("outputs/manga_user_analysis")
    print(f"\n=======================================================")
    print(f"LOADING USER ARTWORK SCENE: {scene_dir}")
    print(f"=======================================================")
    scene, camera, manifest, arrays = load_scene_artifacts(scene_dir)
    print(f"Resolution: {scene.shape[1]}x{scene.shape[0]}")
    print(f"Depth representation: {scene.depth_kind}")
    print(f"Normals present: {scene.normals is not None}")

    # Standard baseline
    source_rig = LightingRig(
        ambient_color=(1.0, 1.0, 1.0),
        ambient_strength=0.75,
        lights=(
            Light(name="source_ambient", kind="directional", color=(1.0, 1.0, 1.0), strength=0.45, direction=(0.0, 0.0, 1.0), casts_shadows=False),
        ),
    )

    settings = EngineSettings(
        renderer_mode="auto",
        detail_preservation_mode="f3",
        filmic_highlight_rolloff=True,
    )

    # 1. Point Light on LEFT side of face - RED
    rig_point_red = LightingRig(
        ambient_color=(0.10, 0.03, 0.03),
        ambient_strength=0.06,
        lights=(
            Light(
                name="point_light_left_face_red",
                kind="point",
                position=(-1.80, 0.18, 1.90),  # Positioned directly to the left of the face profile
                strength=3.6,
                color=(1.0, 0.05, 0.08),       # Pure saturated red
                radius=0.10,
                casts_shadows=True,
            ),
        ),
    )

    # 2. Directional Light from the LEFT - RED (raking side light across facial contours)
    rig_dir_red = LightingRig(
        ambient_color=(0.08, 0.02, 0.02),
        ambient_strength=0.05,
        lights=(
            Light(
                name="directional_light_left_red",
                kind="directional",
                direction=(-0.92, 0.10, -0.38), # Vector pointing toward light on the left
                strength=2.2,
                color=(1.0, 0.04, 0.06),
                casts_shadows=True,
            ),
        ),
    )

    # 3. Point Light on LEFT side of face - NEUTRAL STUDIO WHITE
    rig_point_white = LightingRig(
        ambient_color=(0.08, 0.08, 0.10),
        ambient_strength=0.06,
        lights=(
            Light(
                name="point_light_left_face_white",
                kind="point",
                position=(-1.80, 0.18, 1.90),
                strength=3.6,
                color=(1.0, 0.95, 0.90),       # Warm neutral white
                radius=0.10,
                casts_shadows=True,
            ),
        ),
    )

    print("\n>>> [ENGINE RUN 1] Point Light from Left (RED)...")
    res_point_red = relight(scene, camera, source_rig, rig_point_red, settings)
    print("    Active renderer:", res_point_red.metrics.get("renderer_mode"))
    print("    Active specular:", res_point_red.metrics.get("active_specular_path"))
    print("    Max target diffuse:", float(np.max(res_point_red.metrics.get("target_diffuse", [0]))))

    print("\n>>> [ENGINE RUN 2] Directional Light from Left (RED)...")
    res_dir_red = relight(scene, camera, source_rig, rig_dir_red, settings)
    print("    Active renderer:", res_dir_red.metrics.get("renderer_mode"))
    print("    Active specular:", res_dir_red.metrics.get("active_specular_path"))

    print("\n>>> [ENGINE RUN 3] Point Light from Left (WHITE)...")
    res_point_white = relight(scene, camera, source_rig, rig_point_white, settings)
    print("    Active renderer:", res_point_white.metrics.get("renderer_mode"))
    print("    Active specular:", res_point_white.metrics.get("active_specular_path"))

    # Convert to BGR for saving
    bgr_orig = cv2.cvtColor((np.clip(scene.rgb_srgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    bgr_point_red = cv2.cvtColor((np.clip(res_point_red.rgb_srgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    bgr_dir_red = cv2.cvtColor((np.clip(res_dir_red.rgb_srgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    bgr_point_white = cv2.cvtColor((np.clip(res_point_white.rgb_srgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    out_point_red = ARTIFACT_DIR / "engine_manga_point_light_left_red.png"
    out_dir_red = ARTIFACT_DIR / "engine_manga_directional_left_red.png"
    out_point_white = ARTIFACT_DIR / "engine_manga_point_light_left_white.png"
    out_panel = ARTIFACT_DIR / "engine_manga_left_light_comparison_panel.png"

    cv2.imwrite(str(out_point_red), bgr_point_red)
    cv2.imwrite(str(out_dir_red), bgr_dir_red)
    cv2.imwrite(str(out_point_white), bgr_point_white)

    # 4-panel comparison
    h, w, _ = bgr_orig.shape
    banner_h = 60
    panel = np.zeros((h + banner_h, w * 4 + 30, 3), dtype=np.uint8)
    panel[:banner_h, :] = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "RELIGHT STUDIO ENGINE: PROVENANCE = 100% PBR SHADING ENGINE (ZERO GENERATIVE AI / ZERO DIFFUSION)", (20, 38), font, 0.72, (255, 255, 255), 2)

    x0 = 0
    panel[banner_h:, x0:x0+w] = bgr_orig
    cv2.putText(panel, "ORIGINAL INPUT", (x0 + 15, banner_h + 35), font, 0.75, (0, 255, 255), 2)

    x1 = w + 10
    panel[banner_h:, x1:x1+w] = bgr_point_white
    cv2.putText(panel, "ENGINE: LEFT POINT (WHITE)", (x1 + 15, banner_h + 35), font, 0.75, (255, 255, 255), 2)

    x2 = 2 * (w + 10)
    panel[banner_h:, x2:x2+w] = bgr_dir_red
    cv2.putText(panel, "ENGINE: LEFT DIRECTIONAL (RED)", (x2 + 15, banner_h + 35), font, 0.75, (0, 140, 255), 2)

    x3 = 3 * (w + 10)
    panel[banner_h:, x3:x3+w] = bgr_point_red
    cv2.putText(panel, "ENGINE: LEFT POINT (RED)", (x3 + 15, banner_h + 35), font, 0.75, (0, 100, 255), 2)

    cv2.imwrite(str(out_panel), panel)
    print(f"\n[OUTPUT SAVED] Point Red: {out_point_red}")
    print(f"[OUTPUT SAVED] Dir Red:   {out_dir_red}")
    print(f"[OUTPUT SAVED] Point Wht: {out_point_white}")
    print(f"[OUTPUT SAVED] Panel:     {out_panel}")

if __name__ == "__main__":
    run_manga_engine()
