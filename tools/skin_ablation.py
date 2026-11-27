#!/usr/bin/env python3
"""Phase 1 Single-Factor Ablation Study on Skin Over-Brightening and Texture Inflation.

Disables each candidate mechanism one at a time relative to the Phase 0 `studio_3d` baseline
and attributes the share of skin brightness gain (`rel_y - orig_y`) and `lap_rms` inflation
(`lap_ratio - 1.0`) across the 3 existing human/skin scenes. Writes `benchmarks/ablation.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relight_engine.contracts import EngineSettings  # noqa: E402
from relight_engine.engine import relight  # noqa: E402
from relight_engine.workflows import load_scene_artifacts  # noqa: E402
from tools.skin_benchmark import SCENE_SPECS, _roi_mask_from_norm, build_benchmark_rigs, compute_roi_metrics  # noqa: E402


@dataclass(frozen=True)
class AblationCase:
    key: str
    label: str
    env_overrides: dict[str, str]
    settings_fn: Callable[[], EngineSettings]


ABLATION_CASES: list[AblationCase] = [
    AblationCase(
        key="baseline_studio3d",
        label="Baseline (`studio_3d()`)",
        env_overrides={},
        settings_fn=lambda: EngineSettings.studio_3d(),
    ),
    AblationCase(
        key="eff_sd_floor",
        label="1. `eff_sd` source floor (remove `source_shading` skin underestimate)",
        env_overrides={"RELIGHT_ABLATE_EFF_SD_FLOOR": "1"},
        settings_fn=lambda: EngineSettings.studio_3d(),
    ),
    AblationCase(
        key="no_source_shading_blend",
        label="1b. Disable `source_shading` blend & SH fit (`blend=0`, `auto_sh=False`)",
        env_overrides={},
        settings_fn=lambda: EngineSettings.studio_3d(source_shading_blend=0.0, auto_sh_source=False),
    ),
    AblationCase(
        key="no_sss_wrap",
        label="2. Disable subsurface wrap (`RELIGHT_ABLATE_NO_SSS_WRAP=1`)",
        env_overrides={"RELIGHT_ABLATE_NO_SSS_WRAP": "1"},
        settings_fn=lambda: EngineSettings.studio_3d(),
    ),
    AblationCase(
        key="no_indirect_bounce",
        label="3. Disable `compute_indirect_bounce` (`indirect_bounce_strength=0`)",
        env_overrides={},
        settings_fn=lambda: EngineSettings.studio_3d(indirect_bounce_strength=0.0),
    ),
    AblationCase(
        key="no_preserve_fill",
        label="4. Disable `preserve_fill_balance` ambient (`preserve_fill_balance=0`)",
        env_overrides={},
        settings_fn=lambda: EngineSettings.studio_3d(preserve_fill_balance=0.0),
    ),
    AblationCase(
        key="additive_texture",
        label="5. Decouple multiplicative texture scaling (`RELIGHT_ABLATE_ADDITIVE_TEXTURE=1`)",
        env_overrides={"RELIGHT_ABLATE_ADDITIVE_TEXTURE": "1"},
        settings_fn=lambda: EngineSettings.studio_3d(),
    ),
    AblationCase(
        key="no_wavelet",
        label="5b. Disable wavelet detail lock (`detail_preservation_mode='baseline'`)",
        env_overrides={},
        settings_fn=lambda: EngineSettings.studio_3d(detail_preservation_mode="baseline"),
    ),
    AblationCase(
        key="no_glare_sep",
        label="6a. Disable residual glare separation (`RELIGHT_ABLATE_NO_GLARE_SEP=1`)",
        env_overrides={"RELIGHT_ABLATE_NO_GLARE_SEP": "1"},
        settings_fn=lambda: EngineSettings.studio_3d(),
    ),
    AblationCase(
        key="no_chroma_anchor",
        label="6b. Disable perceptual color anchor (`RELIGHT_ABLATE_NO_CHROMA_ANCHOR=1`)",
        env_overrides={"RELIGHT_ABLATE_NO_CHROMA_ANCHOR": "1"},
        settings_fn=lambda: EngineSettings.studio_3d(),
    ),
]


def _run_with_env(case: AblationCase, scene, camera, src_rig, tgt_rig, roi_mask: np.ndarray) -> dict[str, float]:
    prev: dict[str, str | None] = {k: os.environ.get(k) for k in case.env_overrides}
    try:
        for k, v in case.env_overrides.items():
            os.environ[k] = v
        res = relight(scene, camera, src_rig, tgt_rig, case.settings_fn())
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
    return compute_roi_metrics(scene.rgb_srgb, res.rgb_srgb, roi_mask)


def run_ablation(outputs_root: Path, md_out: Path) -> str:
    human_specs = [s for s in SCENE_SPECS if s["kind"] == "human_skin"]
    loaded = {}
    for spec in human_specs:
        name = spec["name"]
        scene, camera, _, _ = load_scene_artifacts(outputs_root / name)
        src_rig, tgt_rig = build_benchmark_rigs(scene, camera, spec["rig_mode"])
        roi_mask = _roi_mask_from_norm(scene.shape, spec["roi_norm"])
        loaded[name] = (scene, camera, src_rig, tgt_rig, roi_mask)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for case in ABLATION_CASES:
        results[case.key] = {}
        for spec in human_specs:
            name = spec["name"]
            scene, camera, src_rig, tgt_rig, roi_mask = loaded[name]
            results[case.key][name] = _run_with_env(case, scene, camera, src_rig, tgt_rig, roi_mask)

    base = results["baseline_studio3d"]

    lines: list[str] = [
        "# Phase 1 Skin Photorealism Single-Factor Ablation Report",
        "",
        "## Baseline Error to Attribute (`studio_3d()`)",
        "",
        "| Scene (Skin ROI) | `orig_y` | `relit_y` | Brightness Gain (`ΔY`) | `lap_ratio` | `lap_rms` Inflation (`lap_ratio - 1.0`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for spec in human_specs:
        name = spec["name"]
        m = base[name]
        lines.append(
            f"| `{name}` | `{m['orig_mean_luma']:.3f}` | `{m['relit_mean_luma']:.3f}` | "
            f"`+{m['brightness_delta']:.3f}` | `{m['lap_ratio']:.3f}` | `+{(m['lap_ratio'] - 1.0)*100:.1f}%` |"
        )

    lines.extend([
        "",
        "## Single-Factor Ablation Attribution",
        "",
        "Share of error is computed as `(baseline_err - ablated_err) / baseline_err` (positive = mechanism contributes to error; negative = mechanism suppresses error).",
        "",
        "| Mechanism Ablated | Mean ΔY Share (%) | Mean `lap_rms` Inflation Share (%) | `pinstripe` ΔY / `lap_r` | `portrait` ΔY / `lap_r` | `watch` ΔY / `lap_r` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for case in ABLATION_CASES[1:]:
        row = results[case.key]
        dy_shares = []
        lap_shares = []
        per_scene_cells = []
        for spec in human_specs:
            name = spec["name"]
            b = base[name]
            m = row[name]
            b_dy = b["brightness_delta"]
            m_dy = m["brightness_delta"]
            b_lap_inf = b["lap_ratio"] - 1.0
            m_lap_inf = m["lap_ratio"] - 1.0
            dy_sh = ((b_dy - m_dy) / b_dy * 100.0) if abs(b_dy) > 1e-5 else 0.0
            lap_sh = ((b_lap_inf - m_lap_inf) / b_lap_inf * 100.0) if abs(b_lap_inf) > 1e-5 else 0.0
            dy_shares.append(dy_sh)
            lap_shares.append(lap_sh)
            per_scene_cells.append(f"`+{m_dy:.3f}` / `{m['lap_ratio']:.3f}`")
        lines.append(
            f"| {case.label} | `{np.mean(dy_shares):+.1f}%` | `{np.mean(lap_shares):+.1f}%` | "
            + " | ".join(per_scene_cells)
            + " |"
        )

    report = "\n".join(lines) + "\n"
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    report = run_ablation(ROOT / "outputs", ROOT / "benchmarks" / "ablation.md")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
