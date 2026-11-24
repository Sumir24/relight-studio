# EXP-02 — Multi-domain input and scene-quality validation

Date: 2026-11-24

## Objective

Make the universal scene path testable across arbitrary image inputs and introduce a quality gate that flags obviously degenerate depth, geometry and segmentation outputs before the relighting renderer consumes them.

## Implemented in this iteration

- Added `relight_engine/quality.py` to evaluate structural scene-buffer validity and obvious failure modes.
- Added `scripts/validate_scene_buffers.py` to audit a saved `scene_buffers.npz` and its manifest.
- Integrated `scene_quality_report.json` generation into `scripts/run_scene_analysis.py`.
- Added `scripts/benchmark_image_inputs.py` for arbitrary `LABEL=PATH` raster inputs. Labels are report metadata only; they do not change processing.
- Added `benchmarks/run_multidomain_scene_benchmark.py` to run the same neural scene-analysis CLI over a mixed set of images and collect each scene manifest and quality report.
- Fixed `load_scene_artifacts` to restore the saved `protected_mask` buffer (with backward compatibility for the earlier `mask_protected` name).
- Added regression coverage showing renderer output is unchanged when only scene-domain metadata changes.
- Added checks for depth collapse/quantization, low confidence, missing/degenerate normals, missing geometry confidence, empty/full-frame masks and nearly identical instance masks.

## Execution completed in this environment

- Existing and new automated tests: **59 passed**.
- Python compile checks: passed.
- All three new/updated CLI help smoke checks: passed.
- Input preflight: **9/9 cases decoded and passed structural input checks**:
  - user-supplied landscape photograph (1536 x 1152)
  - portrait, food, animal, still-life/object, vehicle-scene, architecture-texture and monochrome sample fixtures
  - synthetic RGBA cutout to exercise alpha preservation
- Input preflight report: `benchmarks/reports/exp02_input_preflight.json`.

Auxiliary sample photographs were loaded transiently from the local `scikit-image` example-data API for input-decoder QA. They are not included in the distributable project archive. Their use here does not establish image rights for product use, and the fixture labels are QA grouping labels, not audited semantic annotations.

## What this does and does not validate

The nine-case run validates raster decoding, orientation handling, RGB range, grayscale normalization and alpha-channel handling. It does **not** validate neural scene understanding or visual relighting.

The new scene quality auditor checks buffer structure and obvious degeneracy. A structurally valid scene may still have incorrect depth, masks, normals, material estimates or source illumination. The report explicitly disclaims ground-truth accuracy.

Neural inference was not run here: `transformers` is not installed and no model checkpoints are cached in this environment. Therefore there are no real neural depth maps, semantic masks or cross-domain relit renders in this experiment.

## Run on a machine with model dependencies and checkpoints

Input-only preflight:

```bash
python scripts/benchmark_image_inputs.py \
  --case "portrait=/path/to/portrait.jpg" \
  --case "product=/path/to/product.png" \
  --case "interior=/path/to/interior.jpg" \
  --case "animal=/path/to/animal.jpg" \
  --case "landscape=/path/to/landscape.jpg" \
  --out benchmarks/reports/input_preflight.json
```

Actual neural scene-analysis matrix (the depth checkpoint's polarity must be inspected first):

```bash
python benchmarks/run_multidomain_scene_benchmark.py \
  --case "portrait=/path/to/portrait.jpg" \
  --case "product=/path/to/product.png" \
  --case "interior=/path/to/interior.jpg" \
  --case "animal=/path/to/animal.jpg" \
  --case "landscape=/path/to/landscape.jpg" \
  --near-is-high
```

The command uses `--near-is-high` only as an example. Choose exactly one of `--near-is-high` or `--far-is-high` after verifying polarity for the exact Depth Anything checkpoint with `scripts/inspect_depth_polarity.py`; do not copy the example assumption blindly. The runner executes cases sequentially to reduce peak model memory and saves the per-image scene analysis plus `multidomain_scene_benchmark.json`.

## Next engineering milestone

Run the neural matrix on five real photographs with the exact model revisions pinned. Review discovery text, class/instance masks and the structural quality report. Then perform a controlled DA2-Small vs DA3-Small/Base comparison without changing the shared scene-buffer or renderer contracts.
