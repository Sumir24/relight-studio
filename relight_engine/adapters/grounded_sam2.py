from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np

from ..scene_policy import SemanticRoutingPolicy, semantic_key, compose_scene_masks


@dataclass
class DetectionRecord:
    prompt: str
    category: str  # generic semantic key; not a closed scene taxonomy
    box_xyxy: tuple[float, float, float, float]
    detector_score: float
    mask_quality_score: float
    area_fraction: float
    instance_id: str


@dataclass
class SceneSegmentationPrediction:
    """Open-vocabulary masks registered to the source image.

    `masks` is a dynamic mapping: keys come from the actual prompts discovered for
    this image (for example "person", "ceramic_mug", "sofa", or "red_car").
    No fixed scene taxonomy is assumed.
    """
    masks: dict[str, np.ndarray]
    instance_masks: dict[str, np.ndarray]
    lightable_mask: np.ndarray
    protected_mask: np.ndarray
    detections: list[DetectionRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class HuggingFaceGroundedSAM2:
    """Open-vocabulary Grounding DINO + SAM2 mask generation.

    Grounding DINO proposes boxes for the supplied image-specific concepts; SAM2
    refines those boxes into soft masks. Prompts are mandatory so the adapter cannot
    quietly assume that every image contains a landscape.
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

    def _load(self) -> None:
        try:
            import torch
            from transformers import (
                AutoProcessor,
                AutoModelForZeroShotObjectDetection,
                Sam2Model,
                Sam2Processor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and transformers>=4.57.3 to run Grounding DINO + SAM2. "
                "No heuristic segmentation fallback is silently substituted."
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
        self._det_processor = AutoProcessor.from_pretrained(self.detector_id)
        self._det_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.detector_id).to(device).eval()
        self._sam_processor = Sam2Processor.from_pretrained(self.segmenter_id)
        self._sam_model = Sam2Model.from_pretrained(self.segmenter_id).to(device).eval()
        self._torch = torch
        self._device = device

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
        if not prompts or any(not str(p).strip() for p in prompts):
            raise ValueError("Provide image-specific object/region prompts from scene discovery or the user.")
        normalized_prompts = [" ".join(str(p).strip().split()) for p in prompts]
        keys = [semantic_key(p) for p in normalized_prompts]
        if len(set(keys)) != len(keys):
            raise ValueError("Prompts contain labels that collide after normalization; make them distinct.")
        (policy or SemanticRoutingPolicy()).validate()
        if self._sam_model is None:
            self._load()

        from PIL import Image
        torch = self._torch
        image_u8 = np.round(rgb_srgb * 255.0).astype(np.uint8)
        pil = Image.fromarray(image_u8, mode="RGB")
        h, w = image_u8.shape[:2]
        text_labels = [[p.lower() for p in normalized_prompts]]
        det_inputs = self._det_processor(images=pil, text=text_labels, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            det_outputs = self._det_model(**det_inputs)
        det_results = self._det_processor.post_process_grounded_object_detection(
            det_outputs, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )
        det_result = det_results[0]
        boxes = det_result.get("boxes")
        scores = det_result.get("scores")
        labels = det_result.get("text_labels", det_result.get("labels"))
        candidates: list[tuple[str, float, list[float]]] = []
        if boxes is not None and len(boxes) > 0:
            for box, score, label in zip(boxes, scores, labels):
                if isinstance(label, (int, np.integer)) or (hasattr(label, "item") and not isinstance(label, str)):
                    label_index = int(label.item()) if hasattr(label, "item") else int(label)
                    matched_prompt = normalized_prompts[label_index] if 0 <= label_index < len(normalized_prompts) else None
                else:
                    label_text = str(label).lower()
                    exact = [p for p in normalized_prompts if p.lower() == label_text]
                    contains = [p for p in normalized_prompts if p.lower() in label_text or label_text in p.lower()]
                    matched_prompt = exact[0] if exact else (contains[0] if len(contains) == 1 else None)
                if matched_prompt is None:
                    continue
                xyxy = [float(x) for x in box.detach().cpu().tolist()]
                x1, y1, x2, y2 = xyxy
                x1, x2 = np.clip([x1, x2], 0, w)
                y1, y2 = np.clip([y1, y2], 0, h)
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(h * w)
                if area < self.min_area_fraction:
                    continue
                candidates.append((matched_prompt, float(score.detach().cpu()), [x1, y1, x2, y2]))

        selected: list[tuple[str, float, list[float]]] = []
        for prompt in normalized_prompts:
            matches = [c for c in candidates if c[0] == prompt]
            matches.sort(key=lambda c: c[1], reverse=True)
            selected.extend(matches[: self.max_detections_per_prompt])

        masks_by_label: dict[str, np.ndarray] = {}
        instance_masks: dict[str, np.ndarray] = {}
        instance_counts: dict[str, int] = {}
        detections: list[DetectionRecord] = []
        if selected:
            input_boxes = [[[*box] for _, _, box in selected]]
            sam_inputs = self._sam_processor(images=pil, input_boxes=input_boxes, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                sam_outputs = self._sam_model(**sam_inputs, multimask_output=True)
            processed = self._sam_processor.post_process_masks(
                sam_outputs.pred_masks.detach().cpu(), sam_inputs["original_sizes"], binarize=False
            )[0]
            iou = sam_outputs.iou_scores.detach().cpu()[0]
            for idx, (prompt, detector_score, box) in enumerate(selected):
                label_key = semantic_key(prompt)
                candidate_logits = processed[idx]
                best_idx = int(torch.argmax(iou[idx]).item())
                mask_logits = candidate_logits[best_idx].numpy().astype(np.float32)
                mask_prob = 1.0 / (1.0 + np.exp(-np.clip(mask_logits, -30.0, 30.0)))
                quality = float(iou[idx, best_idx].item())
                weighted = np.clip(mask_prob * detector_score * quality, 0.0, 1.0)
                instance_counts[label_key] = instance_counts.get(label_key, 0) + 1
                instance_id = f"{label_key}__{instance_counts[label_key]:03d}"
                instance_masks[instance_id] = weighted
                if label_key in masks_by_label:
                    masks_by_label[label_key] = np.maximum(masks_by_label[label_key], weighted)
                else:
                    masks_by_label[label_key] = weighted
                x1, y1, x2, y2 = box
                detections.append(DetectionRecord(
                    prompt=prompt, category=label_key, box_xyxy=(x1, y1, x2, y2),
                    detector_score=detector_score, mask_quality_score=quality,
                    area_fraction=float(np.mean(mask_prob > 0.5)), instance_id=instance_id,
                ))

        lightable, protected = compose_scene_masks(masks_by_label, shape=(h, w), policy=policy)
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
                "semantic_policy": (policy or SemanticRoutingPolicy()).to_metadata(),
                "material_maps_estimated": False,
                "status": status,
                "no_detection_behavior": "all_pixels_remain_candidates_unless_explicitly_protected",
            },
        )
