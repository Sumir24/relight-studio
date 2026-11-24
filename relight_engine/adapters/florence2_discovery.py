from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


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
                 max_new_tokens: int = 1024, max_prompts: int = 32) -> None:
        if max_new_tokens < 32 or max_prompts < 1:
            raise ValueError("max_new_tokens must be >=32 and max_prompts >=1.")
        self.model_id = model_id
        self.device_request = device
        self.max_new_tokens = max_new_tokens
        self.max_prompts = max_prompts
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
        if self.device_request == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device_request
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        self._torch, self._device = torch, device

    def _run_task(self, pil_image, task: str) -> dict:
        torch = self._torch
        inputs = self._processor(text=task, images=pil_image, return_tensors="pt")
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
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        return self._processor.post_process_generation(
            text, task=task, image_size=(pil_image.width, pil_image.height)
        )

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
        dense_result = self._run_task(pil, "<DENSE_REGION_CAPTION>")
        caption = str(caption_result.get("<DETAILED_CAPTION>", "")).strip()
        dense = dense_result.get("<DENSE_REGION_CAPTION>", {})
        labels = dense.get("labels", []) if isinstance(dense, dict) else []
        boxes = dense.get("bboxes", []) if isinstance(dense, dict) else []
        prompts: list[str] = []
        regions: list[dict] = []
        seen: set[str] = set()
        for index, label in enumerate(labels):
            phrase = " ".join(str(label).strip().split())
            key = phrase.casefold()
            if not phrase or key in seen:
                continue
            seen.add(key)
            prompts.append(phrase)
            region = {"label": phrase}
            if index < len(boxes):
                region["box_xyxy"] = [float(v) for v in boxes[index]]
            regions.append(region)
            if len(prompts) >= self.max_prompts:
                break
        if not prompts:
            raise RuntimeError(
                "Florence-2 did not return dense region labels for this image. "
                "Provide explicit --prompts or review the image manually; do not insert scene-specific defaults."
            )
        return SceneDiscoveryResult(
            caption=caption, prompts=tuple(prompts), regions=regions,
            metadata={"model_id": self.model_id, "device": self._device,
                      "task_sequence": ["<DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>"],
                      "license_note": "Florence-2 model card identifies MIT license; verify exact checkpoint at deployment.",
                      "labels_are_candidate_descriptions_not_ground_truth": True},
        )
