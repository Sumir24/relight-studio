# Phase 1 Skin Photorealism Single-Factor Ablation Report

## Baseline Error to Attribute (`studio_3d()`)

| Scene (Skin ROI) | `orig_y` | `relit_y` | Brightness Gain (`ΔY`) | `lap_ratio` | `lap_rms` Inflation (`lap_ratio - 1.0`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `pinstripe_two_lights_mps` | `0.449` | `0.569` | `+0.120` | `1.163` | `+16.3%` |
| `portrait_red_point_mps` | `0.455` | `0.571` | `+0.116` | `1.149` | `+14.9%` |
| `watch_point_mps` | `0.364` | `0.474` | `+0.110` | `1.434` | `+43.4%` |

## Single-Factor Ablation Attribution

Share of error is computed as `(baseline_err - ablated_err) / baseline_err` (positive = mechanism contributes to error; negative = mechanism suppresses error).

| Mechanism Ablated | Mean ΔY Share (%) | Mean `lap_rms` Inflation Share (%) | `pinstripe` ΔY / `lap_r` | `portrait` ΔY / `lap_r` | `watch` ΔY / `lap_r` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. `eff_sd` source floor (remove `source_shading` skin underestimate) | `+3.8%` | `+2.5%` | `+0.120` / `1.162` | `+0.116` / `1.149` | `+0.097` / `1.403` |
| 1b. Disable `source_shading` blend & SH fit (`blend=0`, `auto_sh=False`) | `+15.2%` | `+16.7%` | `+0.070` / `1.096` | `+0.131` / `1.143` | `+0.091` / `1.410` |
| 2. Disable subsurface wrap (`RELIGHT_ABLATE_NO_SSS_WRAP=1`) | `-0.0%` | `-0.0%` | `+0.120` / `1.163` | `+0.116` / `1.149` | `+0.110` / `1.434` |
| 3. Disable `compute_indirect_bounce` (`indirect_bounce_strength=0`) | `+12.3%` | `+5.7%` | `+0.116` / `1.161` | `+0.094` / `1.114` | `+0.094` / `1.466` |
| 4. Disable `preserve_fill_balance` ambient (`preserve_fill_balance=0`) | `-9.3%` | `-7.0%` | `+0.121` / `1.166` | `+0.132` / `1.174` | `+0.124` / `1.442` |
| 5. Decouple multiplicative texture scaling (`RELIGHT_ABLATE_ADDITIVE_TEXTURE=1`) | `-5.9%` | `+47.6%` | `+0.137` / `1.038` | `+0.119` / `1.083` | `+0.110` / `1.340` |
| 5b. Disable wavelet detail lock (`detail_preservation_mode='baseline'`) | `+2.0%` | `+31.1%` | `+0.114` / `1.037` | `+0.115` / `1.081` | `+0.110` / `1.563` |
| 6a. Disable residual glare separation (`RELIGHT_ABLATE_NO_GLARE_SEP=1`) | `-1.9%` | `-3.5%` | `+0.123` / `1.175` | `+0.119` / `1.151` | `+0.111` / `1.440` |
| 6b. Disable perceptual color anchor (`RELIGHT_ABLATE_NO_CHROMA_ANCHOR=1`) | `+17.4%` | `+17.9%` | `+0.092` / `1.129` | `+0.088` / `1.109` | `+0.104` / `1.408` |
