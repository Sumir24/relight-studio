# Relight Studio — Universal Core Engine Prototype

Relight Studio is a general-purpose 2.5D virtual lighting engine for existing photographs. It is not designed for one landscape image or one fixed scene category. The same renderer must support portraits, products, food, interiors, architecture, landscapes, street scenes, vehicles, animals, macro, still life and other photo types.

## Core rule

**Adaptive scene understanding, shared lighting engine.** Image labels are open-vocabulary data, not a fixed class list. The renderer does not contain special cases for mountains, water, sky, faces, cars, or sofas. All visible pixels are candidates by default; explicit protection masks, geometry/material confidence and user edits control where/how target light is applied.

## What is implemented

### Deterministic lighting core

- Shared camera-aligned scene buffers with strict resolution, range and finite-value validation.
- Camera-space depth back-projection and depth-derived or supplied surface normals.
- Directional, point and sampled rectangular-area lights.
- Lambertian diffuse and GGX microfacet specular response.
- Approximate screen-space 2.5D visibility for cast-shadow occlusion.
- GGX specular response with optional roughness, metalness, dielectric specular-level and RGB conductor F0 maps.
- Optional material-confidence blending toward declared conservative priors, with per-map provenance in scene QA and render metrics.
- Source-to-target shading transfer with detail retention, confidence-aware compositing and highlight protection.

### General-purpose scene understanding path

```text
Any supported raster photograph (RGB/RGBA, common JPEG/PNG/TIFF/WebP formats)
  -> EXIF orientation + ICC-to-sRGB + alpha preservation
  -> Florence-2 detailed caption + dense region labels (automatic discovery)
  -> open-vocabulary region proposals
  -> Grounding DINO + SAM2 masks (dynamic label keys)
  -> relative depth + camera-aligned point/normal buffers
  -> neutral semantic policy + optional explicit protected masks
  -> source-light calibration + generic target light rig
  -> material-aware response + final reconstruction
```

- Florence-2 Base adapter discovers image-specific concepts from dense region descriptions. Neural inference is optional and requires model dependencies/checkpoints.
- Grounding DINO Tiny + SAM 2.1 Hiera Tiny produce soft masks from any supplied/discovered object descriptions.
- Dynamic class masks and separately addressable instance masks are generated from prompts, such as `person` + `person__001`/`person__002`, `red_car__001`, `ceramic_mug__001`, `wooden_table__001`, `dog__001`, or `building__001`.
- Both mask families are retained in the shared scene-buffer object and NPZ artifacts, then validated for alignment and finite [0,1] values.
- Transparent input images keep alpha as an independent mask and restore it on output; the engine does not silently flatten cut-outs onto an arbitrary background.
- No scene category is assumed to exist in every image.
- If discovery fails, the pipeline errors or asks for image-specific labels; it does not silently inject landscape prompts.

### Semantic routing policy

```python
from relight_engine.scene_policy import SemanticRoutingPolicy, compose_scene_masks

# Neutral default: all visible pixels are candidates; no class is hardcoded protected.
lightable, protected = compose_scene_masks(masks, shape=(height, width))

# Explicit user/product rule: protect the face but allow all other regions.
policy = SemanticRoutingPolicy(protected_labels=("face",))
lightable, protected = compose_scene_masks(masks, shape=(height, width), policy=policy)
```

A class such as skin, water, glass or metal is not automatically frozen. Future material estimators may propose a special response, but that behaviour must be evidence-based and reviewable.

## Install

Core-only setup:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Optional neural inference dependencies:

```bash
python -m pip install -r requirements-models.txt
```

The optional model requirements include the `einops` and `timm` packages imported by the official Florence-2 custom implementation.

For repeatable runs, pass a full 40-character Hugging Face commit SHA for each model:

```bash
python scripts/run_scene_analysis.py \
  --image path/to/photo.jpg \
  --out scene_analysis \
  --hfov 60 \
  --near-is-high \
  --depth-revision FULL_DEPTH_COMMIT_SHA \
  --discovery-revision FULL_FLORENCE_COMMIT_SHA \
  --detector-revision FULL_GROUNDING_DINO_COMMIT_SHA \
  --segmenter-revision FULL_SAM2_COMMIT_SHA \
  --strict-model-revisions
```

The `--strict-model-revisions` option rejects missing, branch, tag, or abbreviated revisions. It requires full 40-character commit IDs for every model used by the selected run. If the user supplies `--prompts`, Florence-2 is skipped and does not require a revision. Without strict mode, the runner records whether each model uses the default ref, a named ref, or a full commit SHA in the manifest.

Run the real-checkpoint CPU smoke test (downloads the configured model checkpoints and the documented car example):

```bash
python scripts/run_model_integration_smoke.py
```

This smoke test checks actual inference wiring and sequential cleanup on CPU. It pins immutable model commits recorded in `LICENSE_MODEL_NOTES.md`. It is not an Apple MPS test, depth-polarity validation, multi-domain benchmark, or visual-quality certification.

Install the PyTorch build appropriate for the target device first. Florence-2 uses `trust_remote_code=True`; pin and audit the exact repository revision before deployment. The automatic model stack downloads checkpoints unless already cached.

### User-authored material assignments

Material assignments are applied to dynamic scene masks from `scene_buffers.npz`; the semantic label does not automatically imply a material. For example, add an assignment like this to `material_assignments` in the reviewed relight config:

```json
[
  {
    "name": "warm metal trim",
    "mask_key": "metal_trim",
    "roughness": 0.22,
    "metalness": 1.0,
    "specular_level": 0.04,
    "specular_color": [0.78, 0.52, 0.28],
    "blend": 1.0,
    "confidence": 0.85
  }
]
```

`specular_color` is linear-RGB conductor F0, not an sRGB display colour or a light tint. `blend` controls the mask assignment mix; `confidence` is a separate caller-declared reliability weight. A missing confidence map uses authored map values as provided. The renderer's conservative fallback priors are roughness 0.75, metalness 0.0, dielectric specular level 0.04 and neutral conductor F0 [0.65, 0.65, 0.65]. These are declared fallbacks, not inferred material truth. Material provenance and supported/unsupported response families appear in the QA and render reports.


## Neural inference memory lifecycle

Continuous Integration runs Python 3.14 compilation and unit tests for pull requests targeting `main`. A separate model-integration workflow downloads and executes the real Hugging Face checkpoints on CPU for the model-stack smoke milestone.

The scene-analysis command runs neural stages sequentially and releases each model
after extracting CPU-side results:

1. Depth Anything V2 produces the depth and confidence arrays, then its model is released.
2. Florence-2 produces candidate labels/regions, then its model is released.
3. Grounding DINO produces Python box/score records and is released before SAM 2 loads.
4. SAM 2 produces NumPy masks and is released before geometry/scene-buffer construction.

The scene manifest records an `inference_runtime` section with stage duration,
resolved device, best-effort backend allocator counters before release, and cleanup
results. Grounding DINO/SAM 2 stage-level counters are also recorded in the
segmentation metadata.

**Memory diagnostic limitation:** MPS and CUDA allocator counters are not total
process RAM, unified-memory pressure, or a complete peak-memory profile. CPU-side
arrays, processors, image buffers, Python allocations, and macOS shared-memory
accounting are not fully represented. Treat these values as debugging diagnostics,
not a guarantee that the complete pipeline fits a particular machine.

Adapters expose `close()` for deterministic cleanup and can lazily reload if reused.
The command invokes cleanup in `finally` blocks so failures during inference also
release partially loaded model state where possible.

## Run scene analysis

Automatic region discovery (default; no scene-specific prompts):

```bash
python scripts/run_scene_analysis.py \
  --image path/to/any_photo.jpg \
  --out scene_analysis \
  --hfov 60 \
  --hfov-source assumed \
  --near-is-high
```

The example `60` degree FOV is an assumption, not a measured camera property. Select `--near-is-high` or `--far-is-high` only after inspecting the exact depth checkpoint's output polarity.

Manual region labels can override or supplement automatic discovery:

```bash
python scripts/run_scene_analysis.py \
  --image path/to/photo.jpg \
  --out scene_analysis \
  --hfov 60 \
  --hfov-source assumed \
  --near-is-high \
  --prompts "person" "leather jacket" "glass bottle" \
  --protected-labels "face"
```

By default all visible image pixels are candidates. To restrict lighting to selected labels, pass `--lightable-labels "leather jacket" "glass bottle"`.

Artifacts include registered depth, normals, confidence, arbitrary `mask_<label>.png` class masks, `instance_mask_<label>__NNN.png` object masks, lightable/protected maps, an overlay, `scene_buffers.npz` and a provenance manifest.

## Current model candidates and license screening

- Florence-2 Base: MIT model card; detailed captions and dense-region captions provide per-image concept proposals. Audit/pin custom code.
- Grounding DINO Tiny: official repo Apache-2.0.
- SAM 2.1 Hiera Tiny: official SAM 2 repo identifies checkpoints and code as Apache-2.0.
- Depth Anything V2 Small: current working relative-depth adapter.
- Depth Anything 3 Small/Base: current evaluation target for stronger general monocular geometry; official DA3 repo lists these checkpoints as Apache 2.0. Do not replace DA2 until the exact DA3 adapter has passed multi-domain tests.
- SAM 3: separate SAM License; not a default commercial dependency without legal review.

See `LICENSE_MODEL_NOTES.md`, `ENGINE_V1_PLAN.md` and `MATERIAL_RESPONSE_ROADMAP.md`. The roadmap separates scene categories (portrait/product/interior) from reusable material response families (dielectric/metallic/translucent/transparent/emissive); semantic names alone never assign physical material properties.

## Current limits

- This is still a 2.5D visible-surface renderer, not full 3D reconstruction.
- Relative depth is not metres; physical point/area-light falloff needs metric scale or calibration.
- The input normalizer covers common raster formats supported by Pillow; RAW camera files, multi-page documents and unusual animated formats require dedicated decoders.
- Approximate shadows cannot represent unseen backsides or true overhangs.
- Florence-2 region labels and detector/mask outputs are proposals, not ground truth.
- Source illumination is user-calibrated, not automatically estimated yet.
- Material maps are user-authored or supplied by a future validated estimator; no automatic material inference is claimed. The current renderer accepts RGB conductor F0 and optional material confidence, and reports conservative priors explicitly.
- Glass, mirrors, liquids, emissive objects, skin/hair and complex products need specialist response estimation and domain validation.
- Neural inference and visual quality have not yet been validated across real photographs in this build environment.

## Validation

```bash
python -m compileall -q relight_engine scripts tests
python -m pytest -q
```

The test suite verifies engine math, contracts, no-op/lighting behaviour, generic semantic policy, and synthetic scene-to-render workflow. Synthetic fixtures are not substitutes for real-photo quality benchmarks.

## Multi-domain preflight and scene quality QA

The engine includes two separate validation levels:

1. **Input preflight** checks arbitrary raster files for decode success, orientation handling, RGB normalization, grayscale handling, and alpha preservation. The case label is only report metadata and never changes the renderer.
2. **Scene-buffer QA** checks depth validity/dynamic range, confidence, normal degeneracy, geometry-confidence coverage, mask coverage, and near-duplicate instance masks. This detects obvious structural failures; it does not claim that predicted geometry or masks match ground truth.

Run input preflight:

```bash
python scripts/benchmark_image_inputs.py \
  --case "portrait=/path/to/portrait.jpg" \
  --case "product=/path/to/product.png" \
  --case "interior=/path/to/interior.jpg" \
  --out benchmarks/reports/input_preflight.json
```

Run structural QA on a previously generated scene:

```bash
python scripts/validate_scene_buffers.py \
  --scene-npz scene_analysis/scene_buffers.npz \
  --manifest scene_analysis/scene_manifest.json \
  --out scene_analysis/scene_quality_report.json
```

Run the same neural scene-analysis path over multiple photographs (requires the optional model dependencies and cached/downloadable checkpoints):

```bash
python benchmarks/run_multidomain_scene_benchmark.py \
  --case "portrait=/path/to/portrait.jpg" \
  --case "product=/path/to/product.png" \
  --case "interior=/path/to/interior.jpg" \
  --case "animal=/path/to/animal.jpg" \
  --case "landscape=/path/to/landscape.jpg" \
  --near-is-high \
  --out benchmarks/reports/multidomain_run
```

The example depth-polarity flag is illustrative only. Verify the exact depth checkpoint's output polarity before choosing `--near-is-high` or `--far-is-high`. The scene-analysis CLI also writes `scene_quality_report.json` beside the scene buffers and manifest.

See `EXP02_STATUS.md` and `benchmarks/README.md` for the scope and interpretation of the current benchmark results.
