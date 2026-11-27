#!/usr/bin/env python3
"""Phase 0 Skin & Non-Organic Regression Benchmark Harness.

Renders the 3 existing human/skin scenes and 4 non-organic regression scenes with the
skin fidelity guard off and (when available) on, records ROI metrics, dumps optional
debug maps, and writes `benchmarks/skin_baseline.json`.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relight_engine.calibration import compute_skin_likelihood_map  # noqa: E402
from relight_engine.color import srgb_to_linear  # noqa: E402
from relight_engine.contracts import EngineSettings, Light, LightingRig, SceneBuffers  # noqa: E402
from relight_engine.engine import _effective_source_diffuse, _prepare_working_scene, relight  # noqa: E402
from relight_engine.geometry import place_point_light_at_uv, place_studio_light, prepare_geometry  # noqa: E402
from relight_engine.lighting import compute_indirect_bounce, evaluate_rig  # noqa: E402
from relight_engine.workflows import load_scene_artifacts  # noqa: E402

SCENE_SPECS: list[dict] = [
    {"name": "pinstripe_two_lights_mps", "kind": "human_skin", "rig_mode": "two", "roi_norm": (0.08, 0.18, 0.45, 0.56)},
    {"name": "portrait_red_point_mps", "kind": "human_skin", "rig_mode": "two", "roi_norm": (0.125, 0.185, 0.44, 0.54)},
    {"name": "watch_point_mps", "kind": "human_skin", "rig_mode": "two", "roi_norm": (0.65, 0.85, 0.10, 0.30)},
    {"name": "chrome_sculpture_mps", "kind": "regression", "rig_mode": "three_chrome", "roi_norm": (0.20, 0.90, 0.15, 0.95)},
    {"name": "chrome_star_mps", "kind": "regression", "rig_mode": "three_chrome", "roi_norm": (0.25, 0.75, 0.15, 0.85)},
    {"name": "mountain_hiker_warm_v2", "kind": "regression", "rig_mode": "mountain", "roi_norm": (0.15, 0.85, 0.10, 0.90)},
    {"name": "manga_user_analysis", "kind": "regression", "rig_mode": "two", "roi_norm": (0.10, 0.90, 0.10, 0.90)},
]


def _pos(x) -> tuple[float, float, float]:
    if isinstance(x, dict):
        x = x["position"]
    elif isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], (tuple, list, np.ndarray)):
        x = x[0]
    return (float(x[0]), float(x[1]), float(x[2]))


def build_benchmark_rigs(scene: SceneBuffers, camera, mode: str) -> tuple[LightingRig, LightingRig]:
    src_rig = LightingRig(
        ambient_strength=0.5,
        lights=(Light("src", "directional", direction=(0.0, 0.0, -1.0), strength=0.5, casts_shadows=False),),
    )
    if mode == "three_chrome":
        tgt_rig = LightingRig(
            ambient_color=(0.15, 0.15, 0.18),
            ambient_strength=0.14,
            lights=(
                Light("red_bl", "point", position=_pos(place_point_light_at_uv(scene, camera, 0.12, 0.85)), color=(1.0, 0.12, 0.10), strength=3.4, radius=0.28, casts_shadows=True),
                Light("yel_tm", "point", position=_pos(place_point_light_at_uv(scene, camera, 0.50, 0.10)), color=(1.0, 0.85, 0.15), strength=3.2, radius=0.28, casts_shadows=True),
                Light("blu_r", "point", position=_pos(place_point_light_at_uv(scene, camera, 0.88, 0.50)), color=(0.12, 0.45, 1.0), strength=3.4, radius=0.28, casts_shadows=True),
            ),
        )
    elif mode == "mountain":
        p_sun = _pos(place_studio_light(scene, camera, azimuth_deg=48.0, elevation_deg=24.0))
        tgt_rig = LightingRig(
            ambient_color=(0.35, 0.45, 0.65),
            ambient_strength=0.25,
            lights=(Light("sun_warm", "point", position=p_sun, color=(1.0, 0.68, 0.38), strength=3.2, radius=0.22, casts_shadows=True),),
        )
    else:
        p1 = _pos(place_studio_light(scene, camera, azimuth_deg=-52.0, elevation_deg=18.0))
        p2 = _pos(place_studio_light(scene, camera, azimuth_deg=55.0, elevation_deg=12.0))
        tgt_rig = LightingRig(
            ambient_color=(0.25, 0.27, 0.30),
            ambient_strength=0.18,
            lights=(
                Light("key_warm", "point", position=p1, color=(1.0, 0.72, 0.48), strength=2.9, radius=0.25, casts_shadows=True),
                Light("rim_cool", "point", position=p2, color=(0.35, 0.62, 1.0), strength=2.4, radius=0.22, casts_shadows=True),
            ),
        )
    return src_rig, tgt_rig


def srgb_to_lab(rgb_srgb: np.ndarray) -> np.ndarray:
    """Convert HxWx3 float32 sRGB [0,1] into exact float32 CIELAB (D65)."""
    lin = srgb_to_linear(np.clip(np.asarray(rgb_srgb, dtype=np.float32), 0.0, 1.0))
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = lin @ m.T
    ref_white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xr = xyz / ref_white[None, None, :]
    eps = np.float32(216.0 / 24389.0)
    kappa = np.float32(24389.0 / 27.0)
    f = np.where(xr > eps, np.cbrt(xr), (kappa * xr + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def compute_roi_metrics(orig_srgb: np.ndarray, relit_srgb: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    """Compute Phase 0 ROI metrics: luminance stats, highlight clip %, lap_rms ratio, and CIELAB shifts."""
    m = np.asarray(roi_mask, dtype=bool)
    if not np.any(m):
        raise ValueError("roi_mask must contain at least one pixel.")

    orig = np.clip(np.asarray(orig_srgb, dtype=np.float32), 0.0, 1.0)
    rel = np.clip(np.asarray(relit_srgb, dtype=np.float32), 0.0, 1.0)

    def _srgb_luma(img: np.ndarray) -> np.ndarray:
        return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)

    yo = _srgb_luma(orig)
    yr = _srgb_luma(rel)

    orig_clip = float(np.mean(np.max(orig[m], axis=-1) > 0.98) * 100.0)
    rel_clip = float(np.mean(np.max(rel[m], axis=-1) > 0.98) * 100.0)

    def _lap_rms(y: np.ndarray) -> float:
        lap = y[1:-1, 1:-1] * 4.0 - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
        mm = m[1:-1, 1:-1]
        if not np.any(mm):
            return 0.0
        vals = lap[mm]
        return float(np.sqrt(np.mean(vals * vals)))

    lap_o = _lap_rms(yo)
    lap_r = _lap_rms(yr)
    lap_ratio = float(lap_r / max(lap_o, 1e-6))

    lab_o = srgb_to_lab(orig)[m]
    lab_r = srgb_to_lab(rel)[m]
    d_lab = lab_r - lab_o
    delta_e = float(np.mean(np.linalg.norm(d_lab, axis=-1)))
    c_o = np.hypot(lab_o[:, 1], lab_o[:, 2])
    c_r = np.hypot(lab_r[:, 1], lab_r[:, 2])
    delta_chroma = float(np.mean(c_r - c_o))
    h_o = np.degrees(np.arctan2(lab_o[:, 2], lab_o[:, 1]))
    h_r = np.degrees(np.arctan2(lab_r[:, 2], lab_r[:, 1]))
    dh = ((h_r - h_o + 180.0) % 360.0) - 180.0
    chrom = (c_o > 3.0) | (c_r > 3.0)
    delta_hue_deg = float(np.mean(dh[chrom])) if np.any(chrom) else float(np.mean(dh))

    return {
        "orig_mean_luma": float(np.mean(yo[m])),
        "orig_p95_luma": float(np.percentile(yo[m], 95)),
        "orig_p99_luma": float(np.percentile(yo[m], 99)),
        "orig_clip_pct": orig_clip,
        "relit_mean_luma": float(np.mean(yr[m])),
        "relit_p95_luma": float(np.percentile(yr[m], 95)),
        "relit_p99_luma": float(np.percentile(yr[m], 99)),
        "relit_clip_pct": rel_clip,
        "clip_delta_pp": float(rel_clip - orig_clip),
        "brightness_delta": float(np.mean(yr[m]) - np.mean(yo[m])),
        "orig_lap_rms": lap_o,
        "relit_lap_rms": lap_r,
        "lap_ratio": lap_ratio,
        "lab_delta_e": delta_e,
        "lab_delta_chroma": delta_chroma,
        "lab_delta_hue_deg": delta_hue_deg,
    }


def _roi_mask_from_norm(shape: tuple[int, int], roi_norm: tuple[float, float, float, float]) -> np.ndarray:
    h, w = shape
    y0, y1, x0, x1 = roi_norm
    mask = np.zeros((h, w), dtype=bool)
    mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
    return mask


def _save_gray_png(arr: np.ndarray, path: Path, scale_max: float | None = None) -> None:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        a = np.mean(a, axis=-1)
    hi = float(scale_max) if scale_max is not None else max(float(np.max(a)), 1e-6)
    u8 = (np.clip(a / max(hi, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8).save(path)


def make_settings(*, guard_on: bool) -> EngineSettings:
    s = EngineSettings.studio_3d()
    if hasattr(s, "skin_fidelity"):
        s = replace(s, skin_fidelity=replace(s.skin_fidelity, enabled=guard_on))
    return s


def run_benchmark(outputs_root: Path, json_out: Path, dump_debug_dir: Path | None = None) -> dict:
    records: list[dict] = []
    for spec in SCENE_SPECS:
        name = spec["name"]
        scene_dir = outputs_root / name
        if not scene_dir.is_dir():
            continue
        scene, camera, _, _ = load_scene_artifacts(scene_dir)
        src_rig, tgt_rig = build_benchmark_rigs(scene, camera, spec["rig_mode"])
        roi_mask = _roi_mask_from_norm(scene.shape, spec["roi_norm"])

        s_off = make_settings(guard_on=False)
        t0 = time.perf_counter()
        res_off = relight(scene, camera, src_rig, tgt_rig, s_off)
        dt_off = time.perf_counter() - t0
        metrics_off = compute_roi_metrics(scene.rgb_srgb, res_off.rgb_srgb, roi_mask)

        s_on = make_settings(guard_on=True)
        t1 = time.perf_counter()
        res_on = relight(scene, camera, src_rig, tgt_rig, s_on)
        dt_on = time.perf_counter() - t1
        metrics_on = compute_roi_metrics(scene.rgb_srgb, res_on.rgb_srgb, roi_mask)

        non_skin_max_diff = float(np.max(np.abs(res_on.rgb_srgb - res_off.rgb_srgb)))

        if dump_debug_dir is not None:
            sc_dir = dump_debug_dir / name
            work = _prepare_working_scene(scene, s_off, camera=camera)
            pts, nrm, _ = prepare_geometry(work, camera)
            sd_src, _, _ = evaluate_rig(work, camera, pts, nrm, src_rig, s_off)
            eff_sd = _effective_source_diffuse(work, nrm, sd_src, s_off)
            td, _, _ = evaluate_rig(work, camera, pts, nrm, tgt_rig, s_off)
            bounce = compute_indirect_bounce(work, pts, nrm, td, s_off)
            skin_w = compute_skin_likelihood_map(
                scene.rgb_srgb,
                semantic_masks=scene.semantic_masks,
                depth=work.depth,
                material_maps={"metalness": work.metalness} if work.metalness is not None else None,
            )
            detail_scale = np.ones(scene.shape, dtype=np.float32)
            _save_gray_png(skin_w, sc_dir / "skin_w.png", scale_max=1.0)
            # Save side-by-side overlay of skin_w on original image for Phase 2 gate inspection
            overlay = np.clip(scene.rgb_srgb * (1.0 - 0.45 * skin_w[..., None]) + np.array([0.1, 0.85, 0.35], dtype=np.float32)[None, None, :] * (0.45 * skin_w[..., None]), 0.0, 1.0)
            Image.fromarray((overlay * 255).astype(np.uint8)).save(sc_dir / "skin_overlay.png")
            _save_gray_png(eff_sd, sc_dir / "eff_sd.png", scale_max=1.5)
            _save_gray_png(bounce, sc_dir / "bounce.png", scale_max=0.5)
            _save_gray_png(detail_scale, sc_dir / "detail_scale.png", scale_max=1.0)

        records.append({
            "scene": name,
            "kind": spec["kind"],
            "rig_mode": spec["rig_mode"],
            "roi_norm": list(spec["roi_norm"]),
            "guard_off_dt_s": round(dt_off, 3),
            "guard_on_dt_s": round(dt_on, 3),
            "guard_off": metrics_off,
            "guard_on": metrics_on,
            "non_skin_max_diff": non_skin_max_diff,
        })

    payload = {"schema_version": "1.0", "phase": "phase0_baseline", "scenes": records}
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0 skin photorealism & regression benchmark.")
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "skin_baseline.json")
    parser.add_argument("--dump-debug", type=Path, default=None, help="Optional directory for debug PNG dumps.")
    args = parser.parse_args()
    payload = run_benchmark(args.outputs_root, args.out, dump_debug_dir=args.dump_debug)
    print(f"Wrote {len(payload['scenes'])} scene benchmarks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

