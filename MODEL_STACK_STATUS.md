# Relight Studio — Model Stack V2 Status

Last updated: 2026-11-24

## Branch and review state

- Development branch: `feature/model-stack-smoke-v2`
- Pull request: [#3 — Model Stack V2](https://github.com/Sumir24/relight-studio/pull/3)
- PR base: `feature/model-lifecycle-v2` (stacked on PR #2)
- PR status: open, draft, not merged
- `main` is not modified by this milestone.

## Implemented in this milestone

### Real-checkpoint CPU smoke harness

- `scripts/run_model_integration_smoke.py` downloads the documented Hugging Face car example and exercises the actual model adapters on CPU.
- It runs Depth Anything V2 raw depth inference, Florence-2 dense-region discovery, and Grounding DINO plus SAM 2 mask inference.
- It releases each adapter between stages and checks output shapes, finiteness, detections and sequential model lifecycle metadata.
- It records a JSON report if all three stages complete.
- The harness adds the repository root to `sys.path` before importing `relight_engine`.

### Immutable model revisions used by the smoke harness

The smoke test uses full commit SHAs instead of following mutable `main` references:

| Model | Revision |
|---|---|
| Depth Anything V2 Small HF | `5426e4f0f36572d16453bbda7a8389317b1bef99` |
| Florence-2 Base | `15bd11834572457e1e8e8fec7f25c83283902db2` |
| Grounding DINO Tiny | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| SAM 2.1 Hiera Tiny | `de431c4043854a71d8101e17995dfe596bf101a5` |

Florence-2 uses `trust_remote_code=True`. The exact revision is pinned for the smoke run, but the model repository's MIT license notice and the Apache-2.0 header in its model implementation remain a file-level provenance question to resolve before redistribution or commercial deployment. See `LICENSE_MODEL_NOTES.md`.

### Material Response V1 baseline

- Added optional HxWx3 linear-RGB conductor F0 (`specular_color`) to the shared scene contract and GGX renderer.
- Added optional HxW `material_confidence`, which blends supplied maps toward declared conservative renderer priors.
- Added `material_metadata` for per-map provenance.
- Extended user-authored, mask-based material assignments to accept RGB conductor F0 and a separate confidence weight.
- Preserved these maps and provenance when constructing or reloading scene buffers and when running scene QA.
- Renderer fallback values are roughness 0.75, metalness 0.0, dielectric specular level 0.04, and neutral conductor F0 (0.65, 0.65, 0.65).
- These values are explicit renderer priors, not inferred material truth.
- No automatic material estimator is implemented or claimed.

## Real checkpoint smoke findings

The first checkpoint smoke attempt on GitHub Actions reached actual Depth Anything V2 CPU inference successfully (8.366 seconds on the runner; output shape and finite-value checks passed). Florence-2 then failed during model construction under Transformers 4.57.6 because the pinned remote class did not define `_supports_sdpa`, which Transformers read while selecting SDPA attention.

The Florence adapter now requests `attn_implementation="eager"` explicitly, following the fallback documented by Transformers 4.57.6 for architectures that do not implement SDPA. A unit regression test verifies the eager-attention option and pinned revision are passed to the loader. The updated real-checkpoint smoke job is running to validate the actual checkpoint with this change. The smoke harness now records failures per stage and continues to Grounding DINO/SAM 2 even if Florence fails; the overall job still exits non-zero if any stage failed. The CPU-only GitHub Actions job installs CPU wheels rather than the default CUDA-enabled PyTorch distribution. Grounding DINO and SAM 2 have not yet been reached in a complete successful smoke run.

## Automated validation

- Python: 3.14 on GitHub Actions Linux x86-64.
- `compileall`: passed.
- `pytest`: **93 passed** on the model-stack branch's core CI run.
- This validates the code contracts and deterministic/unit-level behaviour. It does not prove visual relighting quality or Apple MPS compatibility.

## Validation still pending

1. Complete the real-checkpoint CPU smoke run and inspect the generated report.
2. Verify actual checkpoint inference stage by stage, including the exact Transformers API behaviour for Florence-2, Grounding DINO and SAM 2.
3. Run the same model stack on the user's Apple Silicon M5/MPS environment.
4. Review the raw depth polarity output for known near/far objects before using it as geometry.
5. Benchmark at least portrait, product, interior, animal and landscape images before claiming universal scene-quality coverage.
6. Inspect real masks and depth boundaries visually. Unit tests and a successful model download are not substitutes for this review.

## Scope and non-claims

- The CPU smoke test is a one-image integration check, not a multi-domain benchmark.
- The smoke test does not validate near/far polarity or metric depth.
- The scene geometry remains 2.5D visible-surface geometry.
- Material maps remain user-authored or future-estimator inputs; no semantic label is treated as a material.
- MPS allocator counters are not total macOS unified-memory measurements.
- No production-ready or commercial-clearance claim is made by this milestone.
