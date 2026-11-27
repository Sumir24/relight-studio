from __future__ import annotations
from dataclasses import dataclass, field
import re
import numpy as np

from ..scene_policy import semantic_key
from .runtime import memory_snapshot, release_device_cache, release_model_from_device, resolve_torch_device, revision_descriptor

_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_META_PHRASES = frozenset({
    "view", "image", "picture", "photo", "photograph", "scene", "body",
    "background", "foreground", "top", "bottom", "left", "right", "side",
})


@dataclass
class SceneDiscoveryResult:
    caption: str
    prompts: tuple[str, ...]
    regions: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Florence2SceneDiscovery:
    """Optional open-vocabulary scene label discovery for arbitrary photographs.

    Dense region captions provide candidate concepts. They are not treated as masks,
    material truth, or exhaustive ground truth; SAM/DINO and review follow downstream.
    """
    DEFAULT_MODEL = "microsoft/Florence-2-base"

    def __init__(self, model_id: str = DEFAULT_MODEL, *, device: str = "auto",
                 max_new_tokens: int = 1024, max_prompts: int = 32,
                 revision: str | None = None) -> None:
        if max_new_tokens < 32 or max_prompts < 1:
            raise ValueError("max_new_tokens must be >=32 and max_prompts >=1.")
        self.model_id = model_id
        self.device_request = device
        self.max_new_tokens = max_new_tokens
        self.max_prompts = max_prompts
        self.revision = revision
        self._torch = None
        self._processor = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "Automatic scene discovery requires torch and transformers with Florence-2 support. "
                "Install requirements-models.txt; no fixed landscape prompts are substituted."
            ) from exc
        device = resolve_torch_device(torch, self.device_request)
        dtype = torch.float16 if device == "cuda" else torch.float32
        # Retain runtime handles before loading so a partial failure can be cleaned up.
        self._torch, self._device = torch, device
        revision_kwargs = {"revision": self.revision} if self.revision else {}
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, **revision_kwargs
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            **revision_kwargs,
        ).to(device).eval()

    def runtime_snapshot(self) -> dict:
        """Return best-effort allocator counters and loaded-model state."""
        return {
            "device": self._device,
            "model_loaded": self._model is not None,
            "processor_loaded": self._processor is not None,
            "memory": memory_snapshot(self._torch, self._device) if self._torch is not None else {
                "device": self._device, "available": False
            },
        }

    def close(self) -> dict:
        """Release Florence-2 model/processor references and accelerator cache.

        Safe to call repeatedly. A subsequent discover() call lazily reloads the model.
        """
        if self._torch is None:
            return {"released": False, "already_closed": True, "device": self._device}
        before = memory_snapshot(self._torch, self._device)
        was_loaded = self._model is not None or self._processor is not None
        release_model_from_device(self._model)
        self._model = None
        self._processor = None
        cleanup = release_device_cache(self._torch, self._device)
        after = memory_snapshot(self._torch, self._device)
        return {
            "released": was_loaded,
            "device": self._device,
            "memory_before_release": before,
            "memory_after_release": after,
            "cleanup": cleanup,
        }

    def _run_task(self, pil_image, task: str) -> dict:
        torch = self._torch
        # The pinned Florence-2 remote processor batches text and images together.
        # Pass one-item lists so it constructs exactly one prompt for exactly one image.
        inputs = self._processor(
            text=[task], images=[pil_image], return_tensors="pt"
        )
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if getattr(value, "is_floating_point", lambda: False)():
                    dtype = next(self._model.parameters()).dtype
                    moved[key] = value.to(device=self._device, dtype=dtype)
                else:
                    moved[key] = value.to(device=self._device)
            else:
                moved[key] = value
        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=moved["input_ids"], pixel_values=moved["pixel_values"],
                max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=3,
                use_cache=False,
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        post_task = (
            "<CAPTION_TO_PHRASE_GROUNDING>"
            if task.startswith("<CAPTION_TO_PHRASE_GROUNDING>")
            else task
        )
        return self._processor.post_process_generation(
            text, task=post_task, image_size=(pil_image.width, pil_image.height)
        )

    @staticmethod
    def _clean_label(raw_label: str) -> str:
        phrase = " ".join(str(raw_label).strip().split())
        phrase = _LEADING_ARTICLE_RE.sub("", phrase).strip()
        return phrase

    def discover(self, rgb_srgb: np.ndarray) -> SceneDiscoveryResult:
        if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] != 3:
            raise ValueError("rgb_srgb must have shape HxWx3.")
        if not np.isfinite(rgb_srgb).all() or np.any(rgb_srgb < 0) or np.any(rgb_srgb > 1):
            raise ValueError("rgb_srgb must be finite and in [0,1].")
        if self._model is None:
            self._load()
        from PIL import Image
        pil = Image.fromarray(np.round(rgb_srgb * 255).astype(np.uint8), mode="RGB")
        caption_result = self._run_task(pil, "<DETAILED_CAPTION>")
        caption = str(caption_result.get("<DETAILED_CAPTION>", "")).strip()

        task_sequence = ["<DETAILED_CAPTION>"]
        raw_pairs: list[tuple[str, list]] = []

        if caption:
            task_sequence.append("<CAPTION_TO_PHRASE_GROUNDING>")
            pg_result = self._run_task(pil, f"<CAPTION_TO_PHRASE_GROUNDING>{caption}")
            pg = pg_result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
            if isinstance(pg, dict):
                raw_pairs.append((pg.get("labels", []), pg.get("bboxes", [])))

        task_sequence.append("<OD>")
        od_result = self._run_task(pil, "<OD>")
        od = od_result.get("<OD>", {})
        if isinstance(od, dict):
            raw_pairs.append((od.get("labels", []), od.get("bboxes", [])))

        prompts: list[str] = []
        regions: list[dict] = []
        seen_keys: set[str] = set()

        def collect_candidates(labels_list, boxes_list) -> None:
            for index, label in enumerate(labels_list):
                phrase = self._clean_label(label)
                if not phrase or phrase.casefold() in _META_PHRASES:
                    continue
                try:
                    norm_key = semantic_key(phrase)
                except ValueError:
                    continue
                if norm_key in seen_keys:
                    continue
                seen_keys.add(norm_key)
                prompts.append(phrase)
                region = {"label": phrase}
                if index < len(boxes_list):
                    region["box_xyxy"] = [float(v) for v in boxes_list[index]]
                regions.append(region)
                if len(prompts) >= self.max_prompts:
                    return

        for labels_list, boxes_list in raw_pairs:
            collect_candidates(labels_list, boxes_list)
            if len(prompts) >= self.max_prompts:
                break

        if not prompts:
            task_sequence.append("<DENSE_REGION_CAPTION>")
            dense_result = self._run_task(pil, "<DENSE_REGION_CAPTION>")
            dense = dense_result.get("<DENSE_REGION_CAPTION>", {})
            if isinstance(dense, dict):
                collect_candidates(dense.get("labels", []), dense.get("bboxes", []))

        if caption:
            # Grounding/OD sometimes omits body contact surfaces (e.g. "hand", "finger") mentioned in
            # the model's own caption; promote explicitly mentioned contact parts so SAM2 segments them.
            cap_words = set(re.findall(r"[a-z]+", caption.lower()))
            for body_tok in ("hand", "hands", "finger", "fingers", "palm", "wrist", "arm", "face", "skin"):
                if body_tok in cap_words and len(prompts) < self.max_prompts:
                    norm_key = semantic_key(body_tok)
                    if norm_key not in seen_keys:
                        seen_keys.add(norm_key)
                        prompts.append(body_tok)

        if not prompts:
            raise RuntimeError(
                "Florence-2 did not return region labels for this image. "
                "Provide explicit --prompts or review the image manually; do not insert scene-specific defaults."
            )
        return SceneDiscoveryResult(
            caption=caption, prompts=tuple(prompts), regions=regions,
            metadata={
                "model_id": self.model_id,
                "device": self._device,
                **revision_descriptor(self.revision),
                "trust_remote_code": True,
                "remote_code_revision_pinned": revision_descriptor(self.revision)["immutable_commit_pinned"],
                "task_sequence": task_sequence,
                "license_note": "Review the repository LICENSE and each executed remote source file; keep the exact model revision recorded.",
                "labels_are_candidate_descriptions_not_ground_truth": True,
            },
        )
