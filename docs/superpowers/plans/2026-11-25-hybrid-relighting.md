# Model-Agnostic Hybrid Relighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model-agnostic hybrid relighting pipeline combining intrinsic de-lighting and direct normal estimation (`Path 1`) with a physics-steered pluggable neural relighter and deterministic detail/mask lock (`Path 2`).

**Architecture:** Extend `SceneBuffers` with neutral physical intrinsic buffers (`albedo`, `source_shading`, `non_diffuse_residual`), define `@runtime_checkable` adapter `Protocol`s (`NormalEstimator`, `IntrinsicEstimator`, `NeuralRelighter`) with a backend registry, upgrade `engine.py` / `composite.py` with `intrinsic` and `hybrid_neural` rendering + detail lock, and add optional `diffusers`-backed reference adapters (Lotus/Marigold normals, Marigold intrinsics, IC-Light SD1.5 relighter) behind those protocols.

**Tech Stack:** Python 3.14, NumPy, SciPy, OpenCV, Pillow, PyTorch 2.15 (MPS/CUDA/CPU), Transformers `>=4.57.3,<5`, Diffusers `>=0.32`, Accelerate `>=0.34`, pytest.

**Spec:** [2026-11-25-hybrid-relighting-design.md](file:///Users/sumirdayal/Codes/studio/relight_studio_core/docs/superpowers/specs/2026-11-25-hybrid-relighting-design.md)

## Global Constraints

- Python 3.14 compatibility across all modules; core `relight_engine` (`contracts`, `engine`, `composite`, `lighting`, `workflows`, `quality`) must never import `torch`, `transformers`, or `diffusers` at module level.
- Keep `transformers>=4.57.3,<5` pinned in `requirements-models.txt` so Florence-2 remote code does not break.
- Zero vendor/model names (`marigold`, `lotus`, `ic_light`) inside `SceneBuffers`, `EngineSettings`, `engine.py`, `composite.py`, or `quality.py`.
- All neural adapters must support explicit `revision` (40-char commit SHA), `runtime_snapshot()`, and idempotent `close()` using `release_model_from_device` + `release_device_cache`.
- Every existing test in `tests/` (`107 passed`) must remain passing; exact no-op invariance and `protected_mask` / `alpha_mask` preservation must hold across both `intrinsic` and `hybrid_neural` modes.

## Review Focus

- Missing `albedo` or `source_shading` when `renderer_mode="intrinsic"` is explicitly requested must raise a clear `ValueError` instead of silently falling back to ratio mode; `renderer_mode="auto"` falls back cleanly to ratio mode.
- Supplying `renderer_mode="hybrid_neural"` without a `NeuralRelighter` instance must raise a clear `ValueError` rather than silently returning an un-relit image.
- `source_shading` supplied as `HxW` scalar vs `HxWx3` RGB must both validate and broadcast cleanly in `compose_intrinsic`.
- `NeuralRelighter` returning out-of-range, non-finite, or mismatched-resolution arrays must fail fast before detail-lock compositing.
- Reloading `scene_buffers.npz` via `load_scene_artifacts` when `albedo`/`source_shading`/`non_diffuse_residual` were saved must restore those arrays and `intrinsic_metadata` with zero data loss.

---

### Task 1: Model-Neutral Intrinsic Buffers & Settings Contract

**Files:**
- Modify: `relight_engine/contracts.py:30-128,187-235`
- Modify: `relight_engine/adapters/scene_factory.py:8-84`
- Modify: `relight_engine/workflows.py:78-138`
- Modify: `relight_engine/quality.py:192-502`
- Test: `tests/test_contracts.py`, `tests/test_workflows.py`, `tests/test_scene_quality.py`

**Interfaces:**
- Produces on `SceneBuffers`: `albedo: Optional[np.ndarray] = None`, `source_shading: Optional[np.ndarray] = None`, `non_diffuse_residual: Optional[np.ndarray] = None`, `intrinsic_metadata: dict = field(default_factory=dict)`.
- Produces on `EngineSettings`: `renderer_mode: Literal["auto", "ratio", "intrinsic", "hybrid_neural"] = "auto"`, `detail_lock_strength: float = 0.85`.
- Produces on `SemanticPrediction`: `albedo`, `source_shading`, `non_diffuse_residual`, `intrinsic_metadata`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_contracts.py
def test_scene_buffers_validates_intrinsic_maps_and_rejects_bad_shapes_or_negatives():
    scene = make_scene()
    scene.albedo = np.full((*scene.shape, 3), 0.5, dtype=np.float32)
    scene.source_shading = np.full(scene.shape, 0.8, dtype=np.float32)
    scene.non_diffuse_residual = np.zeros((*scene.shape, 3), dtype=np.float32)
    scene.validate()

    bad = make_scene()
    bad.albedo = np.full((*scene.shape, 4), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="albedo"):
        bad.validate()

    bad_shading = make_scene()
    bad_shading.source_shading = np.full(scene.shape, -0.1, dtype=np.float32)
    with pytest.raises(ValueError, match="source_shading"):
        bad_shading.validate()

def test_engine_settings_validates_renderer_mode_and_detail_lock_strength():
    EngineSettings(renderer_mode="intrinsic", detail_lock_strength=0.9).validate()
    with pytest.raises(ValueError, match="renderer_mode"):
        EngineSettings(renderer_mode="unknown").validate()
    with pytest.raises(ValueError, match="detail_lock_strength"):
        EngineSettings(detail_lock_strength=1.5).validate()
```

```python
# tests/test_workflows.py
def test_load_scene_artifacts_restores_intrinsic_buffers_and_metadata(tmp_path):
    # Save scene_buffers.npz with albedo (HxWx3), source_shading (HxW), non_diffuse_residual (HxWx3)
    # and manifest["intrinsic_metadata"], reload via load_scene_artifacts, assert exact values restored.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_contracts.py tests/test_workflows.py -k "intrinsic or renderer_mode" -v`
Expected: FAIL (`TypeError` / missing fields on `SceneBuffers` & `EngineSettings`).

- [ ] **Step 3: Implement intrinsic buffer validation, propagation, persistence loading, and QA reporting**
  - Add `RendererMode = Literal["auto", "ratio", "intrinsic", "hybrid_neural"]` and `albedo`, `source_shading`, `non_diffuse_residual`, `intrinsic_metadata` validation in `relight_engine/contracts.py`.
  - Propagate intrinsic fields through `SemanticPrediction` and `build_scene_buffers` in `relight_engine/adapters/scene_factory.py`.
  - Load `albedo`, `source_shading`, `non_diffuse_residual`, and `intrinsic_metadata` in `load_scene_artifacts` in `relight_engine/workflows.py`.
  - Report `"intrinsics"` availability/provenance in `evaluate_scene_buffers` in `relight_engine/quality.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_contracts.py tests/test_workflows.py tests/test_scene_quality.py -v`
Expected: PASS

---

### Task 2: Pluggable Adapter Protocols & Registry

**Files:**
- Create: `relight_engine/adapters/protocols.py`
- Create: `relight_engine/adapters/registry.py`
- Modify: `relight_engine/adapters/__init__.py`
- Test: `tests/test_adapter_protocols.py`

**Interfaces:**
- Produces in `relight_engine/adapters/protocols.py`:
  - Dataclasses: `NormalPrediction(normals, confidence, metadata)`, `IntrinsicPrediction(albedo, source_shading, non_diffuse_residual=None, roughness=None, metalness=None, specular_level=None, metadata={})`, `NeuralRelightPrediction(rgb_srgb, metadata={})`.
  - `@runtime_checkable` `Protocol`s: `DepthEstimator`, `NormalEstimator`, `IntrinsicEstimator`, `NeuralRelighter`, `SceneDiscoverer`, `SceneSegmenter` (each with `estimate` / `discover` / `render`, `runtime_snapshot()`, and `close()`).
- Produces in `relight_engine/adapters/registry.py`:
  - `register_normal_backend(name, factory)`, `create_normal_backend(name, **kwargs)`
  - `register_intrinsic_backend(name, factory)`, `create_intrinsic_backend(name, **kwargs)`
  - `register_relighter_backend(name, factory)`, `create_relighter_backend(name, **kwargs)`
  - `available_backends() -> dict[str, tuple[str, ...]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_protocols.py
def test_custom_in_memory_adapters_satisfy_protocols_and_registry():
    # Register custom dummy NormalEstimator, IntrinsicEstimator, NeuralRelighter without touching HF/torch
    # Verify isinstance(..., NormalEstimator), create_*_backend(), and unknown backend ValueError.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adapter_protocols.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'relight_engine.adapters.protocols'`).

- [ ] **Step 3: Implement `protocols.py`, `registry.py`, and export in `adapters/__init__.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_adapter_protocols.py -v`
Expected: PASS

---

### Task 3: Intrinsic De-Lighting Renderer & Hybrid Detail-Lock Compositor

**Files:**
- Modify: `relight_engine/composite.py:1-72`
- Modify: `relight_engine/engine.py:20-124`
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: `SceneBuffers` (Task 1), `NeuralRelighter` protocol (Task 2).
- Produces in `relight_engine/composite.py`:
  - `compose_intrinsic(scene, target_diffuse, target_specular, geometry_confidence, settings) -> tuple[np.ndarray, np.ndarray]`
  - `compose_hybrid_neural(scene, neural_rgb_srgb, geometry_confidence, settings) -> np.ndarray`
- Produces in `relight_engine/engine.py`:
  - `relight(scene, camera, source_rig, target_rig, settings=None, *, require_geometry_ready=False, neural_relighter=None) -> RelightResult`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_core.py
def test_intrinsic_mode_lifts_baked_shadow_using_albedo_and_preserves_detail():
    # Create a scene with uniform albedo=0.6, dark baked source_shading=0.15 on left half and 0.9 on right half,
    # rgb_srgb = linear_to_srgb(albedo * source_shading) + high-freq stripe detail.
    # Run relight(..., settings=EngineSettings(renderer_mode="intrinsic")) with frontal target_rig.
    # Assert dark left half lifts toward right half luminance while fine stripe contrast survives and ratio mode differs.

def test_hybrid_neural_mode_locks_high_freq_detail_protected_mask_and_alpha():
    # Inject a fake NeuralRelighter returning a smooth relit image (blurred/altered inside protected region).
    # Run relight(..., settings=EngineSettings(renderer_mode="hybrid_neural"), neural_relighter=fake).
    # Assert protected region stays unchanged, alpha_mask is respected, fine detail transfers onto active region,
    # and missing neural_relighter or non-finite output raises ValueError.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_core.py -k "intrinsic_mode or hybrid_neural_mode" -v`
Expected: FAIL (`TypeError: relight() got an unexpected keyword argument 'neural_relighter'`).

- [ ] **Step 3: Implement `compose_intrinsic`, `compose_hybrid_neural`, and `relight()` routing**
  - `"auto"`: selects `"intrinsic"` when `scene.albedo is not None and scene.source_shading is not None`, else `"ratio"`.
  - `"intrinsic"`: requires `scene.albedo is not None` (and records `"mode": "2.5D intrinsic de-lighting and re-rendering"` in `metrics`).
  - `"hybrid_neural"`: validates `neural_relighter is not None`, computes `target_d, target_s, visibility = evaluate_rig(...)`, calls `neural_relighter.render(scene, target_d, target_s, target_rig, settings)`, validates returned `HxWx3` `[0,1]` finite array, and runs `compose_hybrid_neural` to lock high-frequency log-luminance detail, `lightable_mask`, `alpha_mask`, and `geometry_confidence`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_core.py -v`
Expected: PASS

---

### Task 4: Pluggable Diffusers Normal & Intrinsic Adapters

**Files:**
- Create: `relight_engine/adapters/diffusers_normals.py`
- Create: `relight_engine/adapters/diffusers_intrinsics.py`
- Modify: `relight_engine/adapters/__init__.py`
- Modify: `relight_engine/adapters/registry.py`
- Test: `tests/test_hybrid_adapters.py`

**Interfaces:**
- Produces `DiffusersNormalAdapter` (supporting `Apache-2.0` `jingheya/lotus-normal-g-v1-1` / `lotus` mode and `prs-eth/marigold-normals-v1-1` mode behind `NormalEstimator`, normalizing outputs to unit camera-space `+X right, +Y up, +Z away` facing camera).
- Produces `DiffusersIntrinsicAdapter` (supporting `prs-eth/marigold-iid-lighting-v1-1` / `appearance` modes behind `IntrinsicEstimator`, returning `IntrinsicPrediction` aligned to input `HxW`).
- Registers `"marigold_normals"`, `"marigold_intrinsics_lighting"`, `"marigold_intrinsics_appearance"` in `registry.py`. *(Note: `"lotus_normals"` is deferred to a follow-up PR because `diffusers==0.39.0` does not bundle a built-in Lotus pipeline class without custom remote pipeline code.)*

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hybrid_adapters.py
def test_diffusers_normal_and_intrinsic_adapters_resize_validate_and_close_cleanly(monkeypatch):
    # Mock diffusers pipelines and FakeMPS; verify HxW output alignment, unit-length normals,
    # [0,1] albedo bounds, pinned revision forwarding, and idempotent close() memory release.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hybrid_adapters.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `DiffusersNormalAdapter` and `DiffusersIntrinsicAdapter` and register backends**

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hybrid_adapters.py -v`
Expected: PASS

---

### Task 5: Physics-Steered `NeuralRelighter` Adapter, CLI Integration & Dependencies

**Files:**
- Create: `relight_engine/adapters/ic_light_relighter.py`
- Modify: `relight_engine/adapters/registry.py`
- Modify: `relight_engine/adapters/__init__.py`
- Modify: `relight_engine/workflows.py:197-276`
- Modify: `scripts/run_scene_analysis.py:78-355`
- Modify: `scripts/run_relight_from_scene.py:13-31`
- Modify: `requirements-models.txt`
- Modify: `LICENSE_MODEL_NOTES.md`
- Test: `tests/test_hybrid_adapters.py`, `tests/test_workflows.py`

**Interfaces:**
- Produces `ICLightRelighter` implementing `NeuralRelighter`: converts physics-rendered `target_diffuse + target_specular` into a smooth tone-mapped conditioning illumination map (`bg_source` / initial latent conditioning) aligned to input resolution (padded/resized to multiple of 8 with exact crop/restore back to original `(H, W)`), runs SD1.5 + `lllyasviel/ic-light` weights, releases cleanly via `close()`.
- Registers `"ic_light"` in `registry.py`.
- CLI flags on `scripts/run_scene_analysis.py`: `--normal-backend` (default `"depth_derived"`), `--intrinsic-backend` (default `"none"`), plus `--normal-revision`, `--intrinsic-revision`, saving `albedo`, `source_shading`, `non_diffuse_residual` in `scene_buffers.npz` and `intrinsic_metadata` in `scene_manifest.json`.
- Config / CLI support in `run_scene_relight` & `scripts/run_relight_from_scene.py`: `--renderer` (`auto|ratio|intrinsic|hybrid_neural`), `--relighter` (`none|ic_light` or any registered name).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hybrid_adapters.py & tests/test_workflows.py
def test_ic_light_relighter_builds_physics_shading_map_and_preserves_resolution(monkeypatch):
    # Mock SD1.5 UNet/VAE/tokenizer/text_encoder + safetensors load; verify physics target_diffuse
    # drives the conditioning tensor, output matches odd HxW input shape, and close() frees MPS/CUDA cache.

def test_run_scene_relight_supports_intrinsic_and_registered_relighter(tmp_path):
    # Register a test NeuralRelighter in registry, save a scene with albedo+source_shading,
    # run run_scene_relight with renderer="intrinsic" and renderer="hybrid_neural", assert reports & outputs.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hybrid_adapters.py tests/test_workflows.py -k "ic_light or registered_relighter" -v`
Expected: FAIL

- [ ] **Step 3: Implement `ICLightRelighter`, wire CLI/workflow options, update `requirements-models.txt` and `LICENSE_MODEL_NOTES.md`**

- [ ] **Step 4: Run full verification suite & real MPS end-to-end verification**

Run:
1. `.venv/bin/python -m compileall -q relight_engine scripts tests`
2. `.venv/bin/python -m pytest -v`
3. Install `diffusers>=0.32` & `accelerate>=0.34` in `.venv` (confirming `transformers<5`) and run end-to-end hybrid analysis + relight on `test_inputs/depth_test.jpg` with `--device mps` to visually inspect `relight_intrinsic.png` and `relight_hybrid.png`.
Expected: `0` failures in compileall/pytest, and verified visual comparison on `depth_test.jpg`.

---

## Self-Review

- **Spec coverage:** Neutral `SceneBuffers` intrinsic fields (Task 1), pluggable `Protocol`s & registry (Task 2), deterministic intrinsic + detail-locked hybrid neural rendering (Task 3), optional `diffusers` normal/intrinsic adapters (Task 4), physics-steered `ICLightRelighter` + CLI/deps/license notes (Task 5). No gaps.
- **Type consistency:** `NormalPrediction`, `IntrinsicPrediction`, `NeuralRelightPrediction`, `create_normal_backend`, `create_intrinsic_backend`, `create_relighter_backend`, `compose_intrinsic`, `compose_hybrid_neural`, and `renderer_mode` names match identically across all 5 tasks.
- **Proportion & boundaries:** Core `relight_engine` stays zero-ML-dependency; all `diffusers`/`torch` usage stays isolated inside `relight_engine/adapters/`.
