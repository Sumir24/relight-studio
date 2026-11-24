from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..scene_policy import SemanticRoutingPolicy, semantic_key, compose_scene_masks
from .runtime import memory_snapshot, release_device_cache, resolve_torch_device


@dataclass
class DetectionRecord:
    prompt: str
    category: str
    box_xyxy: tuple[float, float, float, float]
    detector_score: float
    mask_quality_score: float
    area_fraction: float
    instance_id: str


@dataclass
class SceneSegmentationPrediction:
    """Open-vocabulary masks registered to the source image."""
    masks: dict[str, np.ndarray]
    instance_masks: dict[str, np.ndarray]
    lightable_mask: np.ndarray
    protected_mask: np.ndarray
    detections: list[DetectionRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class HuggingFaceGroundedSAM2:
    """Sequential Grounding DINO + SAM2 inference with explicit model release.

    Grounding DINO proposes boxes, then its model is released before SAM2 is loaded.
    The two neural weights are therefore not intentionally resident on the accelerator
    at the same time. Prompts remain image-specific; no closed scene taxonomy is used.
    """
    DEFAULT_DETECTOR = "IDEA-Research/grounding-dino-tiny"
    DEFAULT_SEGMENTER = "facebook/sam2.1-hiera-tiny"

    def __init__(
        self,
        detector_id: str = DEFAULT_DETECTOR,
        segmenter_id: str = DEFAULT_SEGMENTER,
        *,
        device: str = "auto",
        box_threshold: float = 0.28,
        text_threshold: float = 0.22,
        max_detections_per_prompt: int = 3,
        min_area_fraction: float = 0.002,
    ) -> None:
        if not 0 < box_threshold < 1 or not 0 < text_threshold < 1:
            raise ValueError("Detection thresholds must be in (0,1).")
        if max_detections_per_prompt < 1:
            raise ValueError("max_detections_per_prompt must be >= 1.")
        if not 0 <= min_area_fraction < 1:
            raise ValueError("min_area_fraction must be in [0,1).")
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.device_request = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_detections_per_prompt = max_detections_per_prompt
        self.min_area_fraction = min_area_fraction

        self._torch = None
        self._device = None
        self._det_processor = None
        self._det_model = None
        self._sam_processor = None
        self._sam_model = None
        self._lifecycle_stages: list[dict] = []

    def _ensure_runtime(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install PyTorch to use Grounding DINO + SAM2.") from exc
        self._torch = torch
        self._device = resolve_torch_device(torch, self.device_request)

    def _load_detector(self) -> None:
        self._ensure_runtime()
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers>=4.57.3 to run Grounding DINO. "
                "No heuristic segmentation fallback is silently substituted."
            ) from exc
        if self._det_model is None:
            self._det_processor = AutoProcessor.from_pretrained(self.detector_id)
            self._det_model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(self.detector_id)
                .to(self._device)
                .eval()
            )

    def _load_segmenter(self) -> None:
        self._ensure_runtime()
        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise RuntimeError(
                "Install a Transformers version with SAM2 support to generate masks."
            ) from exc
        if self._sam_model is None:
            self._sam_processor = Sam2Processor.from_pretrained(self.segmenter_id)
            self._sam_model = (
                Sam2Model.from_pretrained(self.segmenter_id)
                .to(self._device)
                .eval()
            )

    def _detect_candidates(
        self,
        pil_image,
        normalized_prompts: list[str],
        height: int,
        width: int,
    ) -> list[tuple[str, float, list[float]]]:
        """Run DINO and return only Python box/score values before unloading it."""
        torch = self._torch
        text_labels = [[prompt.lower() for prompt in normalized_prompts]]
        det_inputs = self._det_processor(
            images=pil_image, text=text_labels, return_tensors="pt"
        ).to(self._device)
        with torch.inference_mode():
            det_outputs = self._det_model(**det_inputs)
        det_results = self._det_processor.post_process_grounded_object_detection(
            det_outputs,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )
        det_result = det_results[0]
        boxes = det_result.get("boxes")
        scores = det_result.get("scores")
        labels = det_result.get("text_labels", det_result.get("labels"))
        candidates: list[tuple[str, float, list[float]]] = []

        if boxes is not None and scores is not None and labels is not None and len(boxes) > 0:
            for box, score, label in zip(boxes, scores, labels):
                if isinstance(label, (int, np.integer)) or (
                    hasattr(label, "item") and not isinstance(label, str)
                ):
                    label_index = int(label.item()) if hasattr(label, "item") else int(label)
                    matched_prompt = (
                        normalized_prompts[label_index]
                        if 0 <= label_index < len(normalized_prompts)
                        else None
                    )
                else:
                    label_text = str(label).lower()
                    exact = [
                        prompt for prompt in normalized_prompts
                        if prompt.lower() == label_text
                    ]
                    contains = [
                        prompt for prompt in normalized_prompts
                        if prompt.lower() in label_text or label_text in prompt.lower()
                    ]
                    matched_prompt = exact[0] if exact else (
                        contains[0] if len(contains) == 1 else None
                    )
                if matched_prompt is None:
                    continue
                xyxy = [float(value) for value in box.detach().cpu().tolist()]
                x1, y1, x2, y2 = xyxy
                x1, x2 = np.clip([x1, x2], 0, width)
                y1, y2 = np.clip([y1, y2], 0, height)
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(height * width)
                if area < self.min_area_fraction:
                    continue
                score_value = float(score.detach().cpu().item())
                candidates.append(
                    (matched_prompt, score_value, [float(x1), float(y1), float(x2), float(y2)])
                )

        selected: list[tuple[str, float, list[float]]] = []
        for prompt in normalized_prompts:
            matches = [candidate for candidate in candidates if candidate[0] == prompt]
            matches.sort(key=lambda candidate: candidate[1], reverse=True)
            selected.extend(matches[: self.max_detections_per_prompt])
        return selected

    def _generate_masks(
        self,
        pil_image,
        selected: list[tuple[str, float, list[float]]],
        height: int,
        width: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[DetectionRecord]]:
        """Run SAM2 and return NumPy masks/records before releasing model state."""
        torch = self._torch
        input_boxes = [[box for _, _, box in selected]]
        sam_inputs = self._sam_processor(
            images=pil_image, input_boxes=input_boxes, return_tensors="pt"
        ).to(self._device)
        with torch.inference_mode():
            sam_outputs = self._sam_model(**sam_inputs, multimask_output=True)
        processed = self._sam_processor.post_process_masks(
            sam_outputs.pred_masks.detach().cpu(),
            sam_inputs["original_sizes"],
            binarize=False,
        )[0]
        iou = sam_outputs.iou_scores.detach().cpu()[0]

        masks_by_label: dict[str, np.ndarray] = {}
        instance_masks: dict[str, np.ndarray] = {}
        instance_counts: dict[str, int] = {}
        detections: list[DetectionRecord] = []

        for index, (prompt, detector_score, box) in enumerate(selected):
            label_key = semantic_key(prompt)
            candidate_logits = processed[index]
            best_index = int(torch.argmax(iou[index]).item())
            mask_logits = candidate_logits[best_index].numpy().astype(np.float32)
            mask_prob = 1.0 / (1.0 + np.exp(-np.clip(mask_logits, -30.0, 30.0)))
            quality = float(iou[index, best_index].item())
            weighted = np.clip(mask_prob * detector_score * quality, 0.0, 1.0)
            if weighted.shape != (height, width):
                raise RuntimeError(
                    f"SAM2 returned mask shape {weighted.shape}; expected {(height, width)}."
                )

            instance_counts[label_key] = instance_counts.get(label_key, 0) + 1
            instance_id = f"{label_key}__{instance_counts[label_key]:03d}"
            instance_masks[instance_id] = weighted
            if label_key in masks_by_label:
                masks_by_label[label_key] = np.maximum(masks_by_label[label_key], weighted)
            else:
                masks_by_label[label_key] = weighted

            x1, y1, x2, y2 = box
            detections.append(
                DetectionRecord(
                    prompt=prompt,
                    category=label_key,
                    box_xyxy=(x1, y1, x2, y2),
                    detector_score=detector_score,
                    mask_quality_score=quality,
                    area_fraction=float(np.mean(mask_prob > 0.5)),
                    instance_id=instance_id,
                )
            )
        return masks_by_label, instance_masks, detections

    def _release_component(self, component: str) -> dict:
        torch = self._torch
        if torch is None:
            return {"component": component, "released": False, "device": self._device}
        before = memory_snapshot(torch, self._device)
        if component == "grounding_dino":
            was_loaded = self._det_model is not None or self._det_processor is not None
            self._det_model = None
            self._det_processor = None
        elif component == "sam2":
            was_loaded = self._sam_model is not None or self._sam_processor is not None
            self._sam_model = None
            self._sam_processor = None
        else:
            raise ValueError(f"Unknown model component: {component}")
        cleanup = release_device_cache(torch, self._device)
        after = memory_snapshot(torch, self._device)
        result = {
            "component": component,
            "released": was_loaded,
            "device": self._device,
            "memory_before_release": before,
            "memory_after_release": after,
            "cleanup": cleanup,
            "models_loaded_concurrently": int(
                self._det_model is not None
            ) + int(self._sam_model is not None),
        }
        self._lifecycle_stages.append(result)
        return result

    def runtime_snapshot(self) -> dict:
        return {
            "device": self._device,
            "grounding_dino_loaded": self._det_model is not None,
            "sam2_loaded": self._sam_model is not None,
            "memory": (
                memory_snapshot(self._torch, self._device)
                if self._torch is not None
                else {"device": self._device, "available": False}
            ),
            "lifecycle_stages": list(self._lifecycle_stages),
        }

    def close(self) -> dict:
        """Release any remaining model state and return accelerator diagnostics."""
        if self._torch is None:
            return {
                "released": False,
                "already_closed": True,
                "device": self._device,
                "lifecycle_stages": list(self._lifecycle_stages),
            }
        released = []
        if self._det_model is not None or self._det_processor is not None:
            released.append(self._release_component("grounding_dino"))
        if self._sam_model is not None or self._sam_processor is not None:
            released.append(self._release_component("sam2"))
        if not released:
            cleanup = release_device_cache(self._torch, self._device)
        else:
            cleanup = released[-1]["cleanup"]
        return {
            "released": bool(released),
            "device": self._device,
            "cleanup": cleanup,
            "lifecycle_stages": list(self._lifecycle_stages),
            "max_accelerator_models_loaded_concurrently": 1,
        }

    def estimate(
        self,
        rgb_srgb: np.ndarray,
        prompts: tuple[str, ...] | list[str],
        *,
        policy: SemanticRoutingPolicy | None = None,
    ) -> SceneSegmentationPrediction:
        if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] != 3:
            raise ValueError("rgb_srgb must have shape HxWx3.")
        if not np.isfinite(rgb_srgb).all() or np.any(rgb_srgb < 0) or np.any(rgb_srgb > 1):
            raise ValueError("rgb_srgb must be finite and in [0,1].")
        if not prompts or any(not str(prompt).strip() for prompt in prompts):
            raise ValueError(
                "Provide image-specific object/region prompts from scene discovery or the user."
            )
        normalized_prompts = [" ".join(str(prompt).strip().split()) for prompt in prompts]
        keys = [semantic_key(prompt) for prompt in normalized_prompts]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Prompts contain labels that collide after normalization; make them distinct."
            )
        effective_policy = policy or SemanticRoutingPolicy()
        effective_policy.validate()

        from PIL import Image

        image_u8 = np.round(rgb_srgb * 255.0).astype(np.uint8)
        pil_image = Image.fromarray(image_u8, mode="RGB")
        height, width = image_u8.shape[:2]
        self._lifecycle_stages = []

        # Phase A: detection. The helper returns only ordinary Python values; its
        # tensor locals have gone out of scope before the detector is unloaded.
        try:
            self._load_detector()
            selected = self._detect_candidates(
                pil_image, normalized_prompts, height, width
            )
        finally:
            self._release_component("grounding_dino")

        # Phase B: segmentation. SAM2 is not loaded until Grounding DINO is released.
        if selected:
            try:
                self._load_segmenter()
                masks_by_label, instance_masks, detections = self._generate_masks(
                    pil_image, selected, height, width
                )
            finally:
                self._release_component("sam2")
        else:
            masks_by_label = {}
            instance_masks = {}
            detections = []

        lightable, protected = compose_scene_masks(
            masks_by_label, shape=(height, width), policy=effective_policy
        )
        status = "ok" if detections else "no_detections"
        return SceneSegmentationPrediction(
            masks=masks_by_label,
            instance_masks=instance_masks,
            lightable_mask=lightable,
            protected_mask=protected,
            detections=detections,
            metadata={
                "detector_id": self.detector_id,
                "segmenter_id": self.segmenter_id,
                "device": self._device,
                "prompts": normalized_prompts,
                "box_threshold": self.box_threshold,
                "text_threshold": self.text_threshold,
                "scores_are_calibrated": False,
                "mask_generation": "Grounding DINO open-vocabulary boxes refined by SAM2",
                "semantic_policy": effective_policy.to_metadata(),
                "material_maps_estimated": False,
                "status": status,
                "no_detection_behavior": "all_pixels_remain_candidates_unless_explicitly_protected",
                "model_lifecycle": {
                    "stages": list(self._lifecycle_stages),
                    "max_accelerator_models_loaded_concurrently": 1,
                    "models_loaded_sequentially": True,
                },
            },
        )
