# Model-Agnostic Hybrid Relighting Architecture

**Date:** 2026-11-25
**Status:** Approved for planning

## Intent

Upgrade Relight Studio from a single-pass 2.5D ratio-transfer renderer into a **model-agnostic hybrid relighting system** that:
1. Removes baked-in original shadows and lighting via estimated **intrinsic buffers** (`albedo`, `source_shading`, `non_diffuse_residual`) and **direct surface normals** (`Path 1`).
2. Optionally synthesizes photorealistic cast shadows, indirect bounce light, and specular reflections via a pluggable **neural relighter** steered by the physics rig's rendered shading map (`Path 2`).
3. Preserves fine photographic texture, `protected_mask` regions, and `alpha_mask` transparency deterministically regardless of which neural backend is active.
4. Remains completely decoupled from any specific vendor or model family (`Marigold`, `Lotus`, `IC-Light`, etc.) via `Protocol` interfaces and neutral physical fields on `SceneBuffers`.

## Goals

- Keep `relight_engine` core (`contracts.py`, `engine.py`, `composite.py`, `lighting.py`, `quality.py`, `workflows.py`) free of model-specific names and free of hard dependencies on `torch`, `transformers`, or `diffusers`.
- Enable instant interactive relighting previews using deterministic intrinsic de-lighting (`albedo * target_shading`), plus on-demand photorealistic rendering via a pluggable `NeuralRelighter`.
- Support commercial-clean `Apache-2.0` / `MIT` backends (e.g., Depth Anything V2/3, Grounding DINO, SAM 2; with Apache-2.0 Lotus / MoGe normal plugins planned in a follow-up because `diffusers==0.39.0` omits a built-in Lotus pipeline) alongside optional `Open RAIL-M` / `Open RAIL++-M` reference adapters (`Marigold Normals`, `Marigold IID`, `IC-Light SD1.5`) behind the same protocols.
- Enforce deterministic `close()` / `release_device_cache()` sequential accelerator lifecycle across every new adapter (`max_accelerator_models_loaded_concurrently == 1`).

## Non-Goals

- Full 3D mesh reconstruction or path tracing.
- Coupling `SceneBuffers` or UI config schemas to diffusion-specific concepts like LoRA weights or denoising schedulers inside the core contract.
- Preserving the source photograph by default: `renderer_mode="auto"` uses deterministic source-preserving shading transfer (`ratio` backed by neural normals), while explicit `intrinsic` and `hybrid_neural` modes opt into de-lighting and generative relighting when heavier illumination replacement is desired.

## Architecture Overview

```text
Input Image
  ├── Depth / Discovery / Segmentation  ──► depth, semantic_masks, instance_masks, protected_mask, alpha_mask
  ├── NormalEstimator (Protocol)        ──► normals, normal_confidence
  └── IntrinsicEstimator (Protocol)     ──► albedo, source_shading, non_diffuse_residual, roughness, metalness
                                              │
                                              ▼
                                      SceneBuffers (validated physical grid)
                                              │
                    ┌─────────────────────────┴──────────────────────────┐
                    ▼                                                    ▼
        [Deterministic Path]                                  [Hybrid Neural Path]
  albedo * target_shading + spec + detail              Physics target_shading_map ──► NeuralRelighter (Protocol)
                    │                                                    │
                    └─────────────────────────┬──────────────────────────┘
                                              ▼
                            High-Frequency Detail + Mask Safety Lock
                      (preserves fine texture, protected_mask, alpha_mask)
```

## Key Design Decisions

1. **Neutral Physical Buffers on `SceneBuffers`:** `albedo` (`HxWx3` in `[0,1]`), `source_shading` (`HxWx3` or `HxW`, finite `>= 0`), `non_diffuse_residual` (`HxWx3`, finite `>= 0`), and `intrinsic_metadata: dict`. Missing fields remain `None`.
2. **Protocol-First Adapters (`relight_engine/adapters/protocols.py`):** `NormalEstimator`, `IntrinsicEstimator`, and `NeuralRelighter` are `@runtime_checkable` structural protocols. A lightweight registry (`registry.py`) instantiates configured backends by name without hardcoding imports in `engine.py` or `workflows.py`.
3. **Detail-Lock Post-Composite (`composite.py`):** Any neural relighter output passes through deterministic high-frequency log-luminance detail transfer (`detail_lock_strength`) and `lightable_mask * alpha_mask * geometry_confidence` blending so protected subjects, text, and fine geometry cannot drift.
4. **Strict Dependency Pinning:** `diffusers>=0.32` and `accelerate>=0.34` are optional inference dependencies alongside `transformers>=4.57.3,<5` (preserving Florence-2 compatibility).
