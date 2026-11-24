from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from relight_engine.adapters.depth_anything_hf import HuggingFaceDepthAnythingV2
from relight_engine.adapters.florence2_discovery import Florence2SceneDiscovery
from relight_engine.adapters.grounded_sam2 import (
    DetectionRecord,
    HuggingFaceGroundedSAM2,
)
from relight_engine.adapters.runtime import (
    is_full_commit_sha,
    memory_snapshot,
    release_device_cache,
    resolve_torch_device,
    revision_descriptor,
)


class FakeMPS:
    def __init__(self):
        self.empty_cache_calls = 0
        self.synchronize_calls = 0
        self.call_order = []

    def current_allocated_memory(self):
        return 128

    def driver_allocated_memory(self):
        return 256

    def recommended_max_memory(self):
        return 1024

    def synchronize(self):
        self.synchronize_calls += 1
        self.call_order.append("synchronize")

    def empty_cache(self):
        self.empty_cache_calls += 1
        self.call_order.append("empty_cache")


class FakeTorch:
    def __init__(self, *, cuda_available=False, mps_available=True):
        self.cuda = SimpleNamespace(
            is_available=lambda: cuda_available,
            empty_cache=lambda: None,
        )
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        )
        self.mps = FakeMPS()


def test_resolve_device_prefers_cuda_then_mps_then_cpu():
    assert resolve_torch_device(FakeTorch(cuda_available=True), "auto") == "cuda"
    assert resolve_torch_device(FakeTorch(mps_available=True), "auto") == "mps"
    assert resolve_torch_device(FakeTorch(mps_available=False), "auto") == "cpu"
    assert resolve_torch_device(FakeTorch(), "cpu") == "cpu"


def test_resolve_device_rejects_unavailable_explicit_accelerator():
    with pytest.raises(RuntimeError, match="MPS was requested"):
        resolve_torch_device(FakeTorch(mps_available=False), "mps")
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_torch_device(FakeTorch(cuda_available=False), "cuda")


def test_memory_snapshot_labels_allocator_scope_and_reads_mps_counters():
    snapshot = memory_snapshot(FakeTorch(), "mps")
    assert snapshot["available"] is True
    assert snapshot["current_allocated_bytes"] == 128
    assert snapshot["driver_allocated_bytes"] == 256
    assert snapshot["recommended_max_bytes"] == 1024
    assert "not_total_process_memory" in snapshot["scope"]


def test_release_device_cache_calls_python_gc_and_mps_empty_cache():
    torch = FakeTorch()
    result = release_device_cache(torch, "mps")
    assert result["python_gc_run"] is True
    assert result["device_cache_cleared"] is True
    assert torch.mps.synchronize_calls == 1
    assert torch.mps.empty_cache_calls == 1
    assert torch.mps.call_order == ["synchronize", "empty_cache"]


@pytest.mark.parametrize(
    "adapter",
    [
        HuggingFaceDepthAnythingV2(relative_near_is_high=True),
        Florence2SceneDiscovery(),
    ],
)
def test_depth_and_florence_close_release_models_idempotently(adapter):
    adapter._torch = FakeTorch()
    adapter._device = "mps"
    adapter._model = object()
    adapter._processor = object()

    result = adapter.close()
    assert result["released"] is True
    assert result["memory_before_release"]["available"] is True
    assert result["cleanup"]["device_cache_cleared"] is True
    assert adapter._model is None
    assert adapter._processor is None

    second = adapter.close()
    assert second["released"] is False


def test_grounded_sam_closes_loaded_components_and_is_idempotent():
    adapter = HuggingFaceGroundedSAM2(device="cpu")
    adapter._torch = FakeTorch(mps_available=False)
    adapter._device = "cpu"
    adapter._det_model = object()
    adapter._det_processor = object()

    result = adapter.close()
    assert result["released"] is True
    assert adapter._det_model is None
    assert adapter._det_processor is None
    assert result["lifecycle_stages"][0]["component"] == "grounding_dino"
    assert result["max_accelerator_models_loaded_concurrently"] == 1

    second = adapter.close()
    assert second["released"] is False


def test_grounded_sam_runs_detector_and_segmenter_sequentially():
    adapter = HuggingFaceGroundedSAM2(device="cpu")
    adapter._torch = FakeTorch(mps_available=False)
    adapter._device = "cpu"
    mask = np.zeros((8, 10), dtype=np.float32)
    mask[2:6, 3:7] = 0.9
    record = DetectionRecord(
        prompt="object",
        category="object",
        box_xyxy=(2.0, 1.0, 7.0, 7.0),
        detector_score=0.9,
        mask_quality_score=0.95,
        area_fraction=0.2,
        instance_id="object__001",
    )

    def load_detector():
        assert adapter._sam_model is None
        adapter._det_model = object()
        adapter._det_processor = object()

    def detect_candidates(*_args):
        assert adapter._det_model is not None
        assert adapter._sam_model is None
        return [("object", 0.9, [2.0, 1.0, 7.0, 7.0])]

    def load_segmenter():
        assert adapter._det_model is None
        assert adapter._det_processor is None
        adapter._sam_model = object()
        adapter._sam_processor = object()

    def generate_masks(*_args):
        assert adapter._det_model is None
        assert adapter._sam_model is not None
        return {"object": mask}, {"object__001": mask.copy()}, [record]

    adapter._load_detector = load_detector
    adapter._detect_candidates = detect_candidates
    adapter._load_segmenter = load_segmenter
    adapter._generate_masks = generate_masks

    result = adapter.estimate(
        np.zeros((8, 10, 3), dtype=np.float32),
        prompts=["object"],
    )

    assert result.metadata["model_lifecycle"]["models_loaded_sequentially"] is True
    assert result.metadata["model_lifecycle"][
        "max_accelerator_models_loaded_concurrently"
    ] == 1
    assert [
        stage["component"]
        for stage in result.metadata["model_lifecycle"]["stages"]
    ] == ["grounding_dino", "sam2"]
    assert all(
        stage["models_loaded_concurrently"] == 0
        for stage in result.metadata["model_lifecycle"]["stages"]
    )
    assert adapter._det_model is None
    assert adapter._sam_model is None
    assert len(result.detections) == 1


def test_grounded_sam_releases_partially_loaded_detector_when_load_fails():
    adapter = HuggingFaceGroundedSAM2(device="cpu")
    adapter._torch = FakeTorch(mps_available=False)
    adapter._device = "cpu"

    def partially_load_then_fail():
        adapter._det_processor = object()
        adapter._det_model = object()
        raise RuntimeError("simulated detector load failure")

    adapter._load_detector = partially_load_then_fail
    with pytest.raises(RuntimeError, match="simulated detector load failure"):
        adapter.estimate(
            np.zeros((8, 10, 3), dtype=np.float32),
            prompts=["object"],
        )

    assert adapter._det_processor is None
    assert adapter._det_model is None
    assert adapter.runtime_snapshot()["grounding_dino_loaded"] is False


def test_grounded_sam_releases_partially_loaded_segmenter_when_load_fails():
    adapter = HuggingFaceGroundedSAM2(device="cpu")
    adapter._torch = FakeTorch(mps_available=False)
    adapter._device = "cpu"
    adapter._load_detector = lambda: setattr(adapter, "_det_model", object())
    adapter._detect_candidates = lambda *_args: [
        ("object", 0.9, [1.0, 1.0, 6.0, 6.0])
    ]

    def partially_load_then_fail():
        assert adapter._det_model is None
        adapter._sam_processor = object()
        adapter._sam_model = object()
        raise RuntimeError("simulated SAM2 load failure")

    adapter._load_segmenter = partially_load_then_fail
    with pytest.raises(RuntimeError, match="simulated SAM2 load failure"):
        adapter.estimate(
            np.zeros((8, 10, 3), dtype=np.float32),
            prompts=["object"],
        )

    assert adapter._sam_processor is None
    assert adapter._sam_model is None
    assert adapter.runtime_snapshot()["sam2_loaded"] is False



def test_full_commit_sha_is_distinguished_from_branch_or_short_ref():
    full_sha = "a" * 40
    assert is_full_commit_sha(full_sha) is True
    assert is_full_commit_sha("a" * 7) is False
    assert is_full_commit_sha("main") is False
    assert is_full_commit_sha(None) is False


def test_revision_descriptor_marks_default_and_named_refs_as_mutable():
    default = revision_descriptor(None)
    named = revision_descriptor("main")

    assert default["revision_kind"] == "default_model_ref"
    assert default["immutable_commit_pinned"] is False
    assert named["revision_kind"] == "named_ref"
    assert named["immutable_commit_pinned"] is False


def test_revision_descriptor_marks_full_commit_sha_as_immutable_pin():
    revision = "0123456789abcdef0123456789abcdef01234567"
    descriptor = revision_descriptor(revision)

    assert descriptor["revision"] == revision
    assert descriptor["revision_kind"] == "full_commit_sha"
    assert descriptor["immutable_commit_pinned"] is True


def test_model_adapters_accept_explicit_revision_configuration():
    revision = "0123456789abcdef0123456789abcdef01234567"
    depth = HuggingFaceDepthAnythingV2(
        relative_near_is_high=True, revision=revision
    )
    florence = Florence2SceneDiscovery(revision=revision)
    segmentation = HuggingFaceGroundedSAM2(
        detector_revision=revision,
        segmenter_revision=revision,
    )

    assert depth.revision == revision
    assert florence.revision == revision
    assert segmentation.detector_revision == revision
    assert segmentation.segmenter_revision == revision



def test_grounding_dino_postprocessing_receives_the_original_input_ids():
    from contextlib import nullcontext

    class Inputs(dict):
        def to(self, _device):
            return self

    class Box:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [1.0, 2.0, 7.0, 8.0]

    class Score:
        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return 0.91

    input_ids = object()
    seen = {}

    class Processor:
        def __call__(self, **_kwargs):
            return Inputs(input_ids=input_ids, pixel_values=object())

        def post_process_grounded_object_detection(self, outputs, **kwargs):
            seen["outputs"] = outputs
            seen.update(kwargs)
            return [{
                "boxes": [Box()],
                "scores": [Score()],
                "text_labels": ["car"],
            }]

    class Model:
        def __call__(self, **_kwargs):
            return "detector-output"

    adapter = HuggingFaceGroundedSAM2(device="cpu")
    adapter._torch = SimpleNamespace(
        inference_mode=lambda: nullcontext()
    )
    adapter._device = "cpu"
    adapter._det_processor = Processor()
    adapter._det_model = Model()

    selected = adapter._detect_candidates(
        pil_image=object(),
        normalized_prompts=["car"],
        height=10,
        width=12,
    )

    assert seen["outputs"] == "detector-output"
    assert seen["input_ids"] is input_ids
    assert seen["target_sizes"] == [(10, 12)]
    assert seen["threshold"] == adapter.box_threshold
    assert seen["text_threshold"] == adapter.text_threshold
    assert selected == [("car", 0.91, [1.0, 2.0, 7.0, 8.0])]



def test_florence_load_uses_eager_attention_and_pinned_revision(monkeypatch):
    from types import ModuleType
    import sys

    revision = "0123456789abcdef0123456789abcdef01234567"
    seen = {"processor": None, "model": None}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            seen["processor"] = {"model_id": model_id, **kwargs}
            return object()

    class FakeModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            seen["model"] = {"model_id": model_id, **kwargs}
            return FakeModel()

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: False)
        ),
        float16="float16",
        float32="float32",
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.AutoModelForCausalLM = FakeAutoModel

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    adapter = Florence2SceneDiscovery(device="cpu", revision=revision)
    adapter._load()

    assert seen["processor"]["revision"] == revision
    assert seen["processor"]["trust_remote_code"] is True
    assert seen["model"]["revision"] == revision
    assert seen["model"]["trust_remote_code"] is True
    assert seen["model"]["attn_implementation"] == "eager"
    assert seen["model"]["torch_dtype"] == "float32"
    assert adapter.runtime_snapshot()["model_loaded"] is True



def test_florence_pairs_single_prompt_and_image_as_one_item_batches():
    from contextlib import nullcontext

    seen = {}

    class FakeTensor:
        def __init__(self, floating=False):
            self.floating = floating
            self.to_calls = []

        def is_floating_point(self):
            return self.floating

        def to(self, **kwargs):
            self.to_calls.append(kwargs)
            return self

    class FakeProcessor:
        def __call__(self, *, text, images, return_tensors):
            seen["text"] = text
            seen["images"] = images
            seen["return_tensors"] = return_tensors
            return {
                "input_ids": FakeTensor(floating=False),
                "pixel_values": FakeTensor(floating=True),
            }

        def batch_decode(self, generated, skip_special_tokens):
            seen["generated"] = generated
            seen["skip_special_tokens"] = skip_special_tokens
            return ["decoded"]

        def post_process_generation(self, decoded, *, task, image_size):
            seen["post_process"] = (decoded, task, image_size)
            return {task: {"caption": "fixture description"}}

    class FakeModel:
        def parameters(self):
            yield SimpleNamespace(dtype="float32")

        def generate(self, **kwargs):
            seen["generate_kwargs"] = kwargs
            return ["generated_tokens"]

    adapter = Florence2SceneDiscovery(device="cpu")
    adapter._torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    adapter._device = "cpu"
    adapter._processor = FakeProcessor()
    adapter._model = FakeModel()
    image = SimpleNamespace(width=10, height=8)

    result = adapter._run_task(image, "<DETAILED_CAPTION>")

    assert seen["text"] == ["<DETAILED_CAPTION>"]
    assert seen["images"] == [image]
    assert seen["return_tensors"] == "pt"
    assert result == {"<DETAILED_CAPTION>": {"caption": "fixture description"}}
    assert seen["generate_kwargs"]["input_ids"] is not None
    assert seen["generate_kwargs"]["pixel_values"] is not None
    assert seen["generate_kwargs"]["use_cache"] is False


def test_grounded_sam2_preserves_mask_probability_for_policy_masks():
    from contextlib import nullcontext
    from relight_engine.scene_policy import SemanticRoutingPolicy

    class FakeTensor:
        def __init__(self, array):
            self._array = np.asarray(array, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

        def item(self):
            return float(self._array)

        def __getitem__(self, idx):
            return FakeTensor(self._array[idx])

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeSamProcessor:
        def __call__(self, **_kwargs):
            return FakeInputs(original_sizes=[(12, 16)])

        def post_process_masks(self, pred_masks, _original_sizes, *, binarize):
            assert binarize is False
            return pred_masks

    class FakeSamModel:
        def __call__(self, **_kwargs):
            logits = np.full((1, 1, 12, 16), -12.0, dtype=np.float32)
            logits[:, :, 2:10, 4:12] = 12.0
            return SimpleNamespace(
                pred_masks=FakeTensor(logits[None, ...]),
                iou_scores=FakeTensor(np.array([[[0.70]]], dtype=np.float32)),
            )

    adapter = HuggingFaceGroundedSAM2(device="cpu")
    adapter._torch = SimpleNamespace(
        inference_mode=lambda: nullcontext(),
        argmax=lambda tensor: FakeTensor(int(np.argmax(tensor.numpy()))),
    )
    adapter._device = "cpu"
    adapter._load_detector = lambda: None
    adapter._detect_candidates = lambda *_args: [("face", 0.35, [4.0, 2.0, 12.0, 10.0])]
    adapter._load_segmenter = lambda: None
    adapter._sam_processor = FakeSamProcessor()
    adapter._sam_model = FakeSamModel()

    result = adapter.estimate(
        np.full((12, 16, 3), 0.5, dtype=np.float32),
        prompts=["face"],
        policy=SemanticRoutingPolicy(protected_labels=("face",)),
    )

    assert float(result.protected_mask[5, 8]) > 0.99
    assert float(result.lightable_mask[5, 8]) < 0.01
    assert result.detections[0].detector_score == pytest.approx(0.35)
    assert result.detections[0].mask_quality_score == pytest.approx(0.70)


def test_florence_discover_uses_phrase_grounding_and_od_without_article_or_key_collisions():
    adapter = Florence2SceneDiscovery(device="cpu")
    adapter._model = object()
    adapter._processor = object()

    def fake_run_task(_pil, task):
        if task == "<DETAILED_CAPTION>":
            return {"<DETAILED_CAPTION>": "A view of the ocean with a statue and a flower pot."}
        if task.startswith("<CAPTION_TO_PHRASE_GROUNDING>"):
            return {
                "<CAPTION_TO_PHRASE_GROUNDING>": {
                    "labels": ["a view", "the ocean", "a statue", "a flower pot"],
                    "bboxes": [
                        [0.0, 0.0, 10.0, 10.0],
                        [0.0, 0.0, 8.0, 10.0],
                        [2.0, 2.0, 5.0, 5.0],
                        [6.0, 6.0, 9.0, 9.0],
                    ],
                }
            }
        if task == "<OD>":
            return {
                "<OD>": {
                    "labels": ["flower-pot", "statue", "tree"],
                    "bboxes": [
                        [6.0, 6.0, 9.0, 9.0],
                        [2.0, 2.0, 5.0, 5.0],
                        [0.0, 5.0, 3.0, 10.0],
                    ],
                }
            }
        raise AssertionError(f"Unexpected task: {task}")

    adapter._run_task = fake_run_task
    result = adapter.discover(np.full((10, 12, 3), 0.5, dtype=np.float32))
    assert "ocean" in result.prompts
    assert "statue" in result.prompts
    assert "tree" in result.prompts
    assert "view" not in result.prompts
    # "a flower pot" and "flower-pot" normalize to the same semantic_key; only one should be emitted
    assert len([p for p in result.prompts if "flower" in p]) == 1


def test_grounded_sam2_deduplicates_highly_overlapping_same_class_masks():
    from contextlib import nullcontext

    class FakeTensor:
        def __init__(self, array):
            self._array = np.asarray(array, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

        def item(self):
            return float(self._array)

        def __getitem__(self, idx):
            return FakeTensor(self._array[idx])

    class FakeInputs(dict):
        def to(self, _device):
            return self

    class FakeSamProcessor:
        def __call__(self, **_kwargs):
            return FakeInputs(original_sizes=[(12, 16)])

        def post_process_masks(self, pred_masks, _original_sizes, *, binarize):
            return pred_masks

    class FakeSamModel:
        def __call__(self, **_kwargs):
            # Two boxes for the same class producing nearly identical masks
            logits = np.full((1, 2, 1, 12, 16), -12.0, dtype=np.float32)
            logits[0, :, 0, 2:10, 4:12] = 12.0
            iou = np.array([[[0.90], [0.80]]], dtype=np.float32)
            return SimpleNamespace(pred_masks=FakeTensor(logits), iou_scores=FakeTensor(iou))

    adapter = HuggingFaceGroundedSAM2(device="cpu")
    adapter._torch = SimpleNamespace(
        inference_mode=lambda: nullcontext(),
        argmax=lambda tensor: FakeTensor(int(np.argmax(tensor.numpy()))),
    )
    adapter._device = "cpu"
    adapter._load_detector = lambda: None
    adapter._detect_candidates = lambda *_args: [
        ("window", 0.72, [4.0, 2.0, 12.0, 10.0]),
        ("window", 0.45, [4.0, 2.0, 11.5, 10.0]),
    ]
    adapter._load_segmenter = lambda: None
    adapter._sam_processor = FakeSamProcessor()
    adapter._sam_model = FakeSamModel()

    result = adapter.estimate(np.full((12, 16, 3), 0.5, dtype=np.float32), prompts=["window"])
    assert len(result.instance_masks) == 1
    assert len(result.detections) == 1
    assert result.detections[0].detector_score == pytest.approx(0.72)


def test_normalize_relative_depth_handles_sub_epsilon_tails_without_uninitialized_values():
    from relight_engine.adapters.depth_anything_hf import normalize_relative_depth

    arr = np.linspace(1.0, 2.0, 1000, dtype=np.float32).reshape(25, 40)
    lo, hi = np.quantile(arr, [0.02, 0.98])
    # Set extreme tails strictly within (lo - 1e-7, lo) and (hi, hi + 1e-7) to trigger float64 vs float32 rounding edge
    arr[0, 0] = np.float32(lo - 5e-8)
    arr[-1, -1] = np.float32(hi + 5e-8)
    arr[arr < arr[0, 0]] = arr[0, 0]
    arr[arr > arr[-1, -1]] = arr[-1, -1]
    out = normalize_relative_depth(arr, near_is_high=True)
    assert np.isfinite(out).all()
    assert float(out.min()) >= 1.0 and float(out.max()) <= 5.0
