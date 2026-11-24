# Universal Scene Pipeline Update — 2026-11-24

## Architectural correction

The previous implementation was too landscape-specific: its default prompts were `sky`, `mountain`, `forest`, `lake`, `tree`; the mask composer hardcoded mountain/vegetation as lightable and sky/water as protected; and the CLI rejected images without mountain/vegetation detections. This contradicted the product requirement for a general-purpose photo relighting engine.

Those assumptions have been removed from the active scene path.

## Code changes

- Added `relight_engine/scene_policy.py` with free-form semantic label normalization and `SemanticRoutingPolicy`.
- Default semantic behavior: every visible pixel is a candidate; no semantic label is protected unless explicitly configured.
- Added optional `protected_labels` and `lightable_labels` policies. User edits and model labels remain dynamic.
- Rewrote Grounding DINO + SAM2 adapter to accept arbitrary image-specific prompts and produce arbitrary mask names. Removed landscape aliases and fixed canonical masks.
- Added `Florence2SceneDiscovery` adapter to produce a detailed caption and dense-region label prompts automatically.
- Updated scene analysis CLI: auto-discovery is used when `--prompts` is omitted; manual prompts are optional overrides; no scene-specific default prompts are injected.
- Updated output writer to save arbitrary `mask_<label>.png` class maps, separate `instance_mask_<label>__NNN.png` maps, and dynamic overlay colours.
- Extended the shared `SceneBuffers` contract to retain class masks, per-instance masks and the protected mask, with shape/range/finite-value validation.
- Added `scene_domains.py` for QA domain hints that explicitly do not alter renderer behaviour.
- Added general raster input normalization with EXIF orientation, ICC-to-sRGB conversion and separate alpha preservation; transparent pixels are protected in compositing and alpha is restored in output.
- Updated README and engine plan to define a single shared renderer plus adaptive scene understanding.

## Current supported QA matrix

Portrait, group photo, product, food, interior, architecture, landscape, street, vehicle, animal, macro, still life, artwork and miscellaneous/mixed scenes.

The table is a coverage plan, not a claim that every domain has already been visually validated.

## Model screening

- Florence-2 Base: official model card lists MIT and describes detailed captioning, object detection, dense-region captions, phrase grounding and region proposals. Its remote-code path must be pinned and audited.
- Grounding DINO official repository: Apache-2.0.
- SAM 2 official repository: checkpoints and code Apache-2.0.
- Depth Anything V2 Small remains the implemented depth adapter.
- Depth Anything 3 Small/Base: official repository lists Apache-2.0 checkpoints and supports general visual geometry; evaluate against DA2 in multi-domain cases before replacing.
- SAM 3: uses a separate SAM License; not selected as a default commercial dependency without legal review.

## Validation in this environment

- Generic semantic-mask tests now cover arbitrary labels: person/face/clothing, product objects, furniture, animals, vehicles and other nouns.
- Input/output tests verify RGB decoding and preservation of transparent PNG alpha.
- Instance-level regression tests verify two separate objects of one class survive the class-mask aggregation and remain addressable through the scene buffer and saved scene artifacts.
- The same mask composer is tested with no detections, with explicit protected regions and explicit lightable allow-lists.
- Existing synthetic engine and workflow tests remain required.
- The common SceneBuffers contract now retains dynamic class and per-instance masks, not only a pre-composed lightable mask.
- The material roadmap separates reusable surface-response families from scene-category labels.
- Current core suite result before EXP-02: 53 tests passed; Python compile check and CLI help check passed.
- EXP-02 adds `relight_engine/quality.py`, scene-quality report integration, an arbitrary-image input preflight CLI, and a multi-image neural scene-analysis benchmark runner.
- EXP-02 fixes protected-mask deserialization: `load_scene_artifacts` now reads the `protected_mask` array written by the current scene-analysis CLI and retains backward compatibility with `mask_protected`.
- Renderer regression test confirms changing only scene-domain metadata (portrait vs vehicle) leaves the shared renderer output unchanged.
- Current complete suite after EXP-02: 59 tests passed.
- Input preflight completed 9/9 temporary test cases, including the original user landscape, portrait/food/animal/object/vehicle/architecture/monochrome sample images, and a synthetic RGBA cutout.
- The input report is `benchmarks/reports/exp02_input_preflight.json`; a synthetic scene-buffer QA report is `benchmarks/reports/exp02_synthetic_scene_quality.json`.
- Actual neural inference is still pending: `transformers` is not installed and no selected model checkpoints are cached. The nine-case result validates image decoding/normalization/alpha paths only, not depth, masks or relighting.
- Automatic neural inference with Florence-2, Grounding DINO, SAM2 and real depth models has not been run in this environment; `transformers` is not installed here. No actual photo relighting result or multi-domain visual validation is claimed.

## Next milestones

1. Use `benchmarks/run_multidomain_scene_benchmark.py` to run one neural benchmark with at least portrait, product, interior, animal and landscape photographs.
2. Review auto-discovered labels and segmentation masks for all cases, including thin structures and overlapping instances.
3. Benchmark Depth Anything 3 Small/Base vs Depth Anything V2 Small across all domains.
4. Add robust material-specific response estimation; the semantic label alone must not be treated as material truth.
5. Improve automatic source-light estimation and confidence reporting.
6. Validate final relighting visually against unchanged input and manually annotated reference regions.
