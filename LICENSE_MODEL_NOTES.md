# Third-party model licence and architecture notes — checked 2026-11-24

These are engineering screening notes, not legal advice. Re-check the exact checkpoint revision, model card, code terms, dependencies and training-data provenance before commercial distribution.

| Candidate | Official evidence found | Engineering decision |
|---|---|---|
| Florence-2 Base (`microsoft/Florence-2-base`) | Hugging Face model card and repository `LICENSE` file list MIT. The current `modeling_florence2.py` source file carries an Apache-2.0 header. Tasks include detailed caption, object detection, dense-region caption, region proposal and phrase grounding. The model uses `trust_remote_code=True`. | Candidate for per-image scene discovery and adaptive prompt generation. Treat the differing repository/file notices as a provenance item to resolve; pin and audit the exact remote code before deployment or redistribution. Not proof of exhaustive discovery. |
| Grounding DINO Tiny (`IDEA-Research/grounding-dino-tiny`) | Official repository lists Apache-2.0. | Candidate open-vocabulary detection stage; labels come from image discovery/user prompts, not a fixed taxonomy. |
| SAM 2.1 Hiera Tiny (`facebook/sam2.1-hiera-tiny`) | Official Meta SAM 2 repository says checkpoints and code are Apache-2.0. | Candidate to turn region proposals into soft pixel masks. |
| Depth Anything V2 Small HF relative | Prior checked model card lists Apache-2.0. | Current implemented baseline depth model, relative depth only. |
| Depth Anything 3 Small / Base | Official DA3 repository lists DA3-SMALL and DA3-BASE checkpoints as Apache 2.0. DA3 provides single-view geometry and multi-view/pose functions. | Evaluation target against DA2 across portrait/product/interior/landscape before replacing current adapter. |
| Depth Anything 3 Metric Large / Mono Large | Official DA3 repository lists DA3METRIC-LARGE and DA3MONO-LARGE as Apache 2.0. | Candidate metric/monocular estimators; test model memory, domain generalization and calibration. |
| Depth Anything 3 Giant/Large / Nested Giant-Large | Official DA3 table lists these original larger variants as CC BY-NC 4.0. | Excluded from commercial default path unless independently licensed. |
| Marigold Normals / Intrinsics (`prs-eth/marigold-normals-v1-1`, `prs-eth/marigold-iid-lighting-v1-1`, `prs-eth/marigold-iid-appearance-v1-1`) | Official Hugging Face model cards list CreativeML Open RAIL++-M (`openrail++`). | Optional pluggable `NormalEstimator` / `IntrinsicEstimator` backends behind model-neutral `SceneBuffers` interfaces; commercial use permitted subject to Open RAIL++-M Appendix A terms. |
| IC-Light SD1.5 (`lllyasviel/ic-light` `iclight_sd15_fc.safetensors` over `stable-diffusion-v1-5/stable-diffusion-v1-5`) | Official `lllyasviel/IC-Light` code is Apache-2.0; SD1.5 weights follow CreativeML Open RAIL-M (`openrail`). *(IC-Light V2 Flux weights are non-commercial and excluded.)* | Optional pluggable `NeuralRelighter` backend; keep behind `NeuralRelighter` protocol so Apache-2.0 or custom models can replace it cleanly. |
| SAM 3 | Official repository uses a separate SAM License, not Apache-2.0. | Not a default commercial dependency without legal review of the full agreement. |
| SegFormer B0 ADE20K (`nvidia/segformer-b0-finetuned-ade-512-512`) | Prior check found model page `other` and source license restrictions for non-commercial research/evaluation. | Excluded from the commercial product path unless new exact terms are cleared. |


## Immutable revisions used by the model-stack smoke test

The CPU smoke harness pins these exact Hugging Face model revisions (full commit SHAs) so the test does not follow a moving `main` branch:

| Model | Repository revision |
|---|---|
| Depth Anything V2 Small HF | `5426e4f0f36572d16453bbda7a8389317b1bef99` |
| Florence-2 Base | `15bd11834572457e1e8e8fec7f25c83283902db2` |
| Grounding DINO Tiny | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| SAM 2.1 Hiera Tiny | `de431c4043854a71d8101e17995dfe596bf101a5` |

These pins make the smoke test repeatable and preserve the exact Florence remote-code revision inspected. They do not settle the repository/file licence-scope question above.

## Architecture decision

Do not rely on one closed category list or one scene-specific segmentation model. Use a small number of model adapters with explicit output contracts:

1. Florence-2 Base proposes labels/regions for the current image.
2. Grounding DINO proposes boxes from those labels or explicit user prompts.
3. SAM 2 refines proposals into dynamic soft masks.
4. Depth estimator produces geometry with units and confidence.
5. The renderer consumes a common scene representation, independent of semantic names.

The first-stage labels can miss objects; all mask/detection scores remain uncalibrated ranking/quality signals unless independently calibrated.

## Florence-2 provenance follow-up

The Hub repository's `LICENSE` file is MIT, while the current `modeling_florence2.py` file contains an Apache-2.0 header. This note records the observed notices; it does not determine which terms govern each component. Before shipping, retain the exact model revision, inspect all executed remote files and their notices, and obtain a legal interpretation if the component scope remains ambiguous.

## Primary sources

- Florence-2 Base model card and tasks: https://huggingface.co/microsoft/Florence-2-base
- Grounding DINO official repository: https://github.com/IDEA-Research/GroundingDINO
- SAM 2 official repository and licensing: https://github.com/facebookresearch/sam2
- Depth Anything 3 official repository/model table: https://github.com/ByteDance-Seed/Depth-Anything-3
- SAM 3 official repository and custom license: https://github.com/facebookresearch/sam3
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
