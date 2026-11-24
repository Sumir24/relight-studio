from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

from .runtime import memory_snapshot, release_device_cache, release_model_from_device, resolve_torch_device, revision_descriptor

@dataclass
class DepthPrediction:
    depth: np.ndarray
    confidence: np.ndarray
    kind: Literal['metric', 'relative']
    metadata: dict = field(default_factory=dict)


def normalize_relative_depth(raw_score: np.ndarray, *, near_is_high: bool,
                             near_value: float = 1.0, far_value: float = 5.0) -> np.ndarray:
    """Convert a relative model score into positive camera-Z-like units.

    The returned numbers are normalized relative units, not metres. `near_is_high`
    must be selected only after checking the exact checkpoint's output polarity.
    """
    raw = np.asarray(raw_score, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("raw_score must be a 2D array.")
    if not np.isfinite(raw).all():
        raise ValueError("raw_score must contain finite values.")
    if not (np.isfinite([near_value, far_value]).all() and 0 < near_value < far_value):
        raise ValueError("Require finite 0 < near_value < far_value.")
    lo, hi = np.quantile(raw, [0.02, 0.98])
    if hi - lo < 1e-6:
        lo = float(np.min(raw))
        hi = float(np.max(raw))
        if hi - lo < 1e-6:
            raise ValueError("Relative depth prediction has insufficient dynamic range.")
    raw_min = float(np.min(raw))
    raw_max = float(np.max(raw))
    # Preserve strictly monotonic gradients in the extreme 2% tails instead of flat clipping.
    low_anchor = 0.04 if raw_min < lo - 1e-7 else 0.0
    high_anchor = 0.96 if raw_max > hi + 1e-7 else 1.0
    unit = np.zeros_like(raw, dtype=np.float32)
    mid_mask = (raw >= lo) & (raw <= hi)
    unit[mid_mask] = low_anchor + (high_anchor - low_anchor) * (raw[mid_mask] - lo) / (hi - lo)
    low_mask = raw < lo
    if low_anchor > 0.0:
        unit[low_mask] = low_anchor * (raw[low_mask] - raw_min) / max(float(lo - raw_min), 1e-8)
    else:
        unit[low_mask] = 0.0
    high_mask = raw > hi
    if high_anchor < 1.0:
        unit[high_mask] = high_anchor + (1.0 - high_anchor) * (raw[high_mask] - hi) / max(float(raw_max - hi), 1e-8)
    else:
        unit[high_mask] = 1.0
    unit = np.clip(unit, 0.0, 1.0)
    if near_is_high:
        unit = 1.0 - unit
    return (near_value + (far_value - near_value) * unit).astype(np.float32)


def relative_depth_polarity_candidates(
    raw_score: np.ndarray, *, near_value: float = 1.0, far_value: float = 5.0
) -> dict[str, np.ndarray]:
    """Return both explicit polarity interpretations for visual checkpoint review.

    The keys describe the assumption about the raw checkpoint score, not an inferred
    truth. A human must inspect known near/far objects and select one interpretation.
    """
    return {
        "near_is_high": normalize_relative_depth(
            raw_score, near_is_high=True, near_value=near_value, far_value=far_value
        ),
        "far_is_high": normalize_relative_depth(
            raw_score, near_is_high=False, near_value=near_value, far_value=far_value
        ),
    }


class HuggingFaceDepthAnythingV2:
    """Optional Transformers adapter for Depth Anything V2.

    The relative Small checkpoint is the commercial-license-conscious starting point.
    For relative depth, the near/far polarity must be explicitly configured after checking
    the selected checkpoint's outputs. This adapter normalizes it into positive camera-Z
    units in [1, 5]; those are relative units, not metres.

    Metric checkpoint loading is blocked unless the caller explicitly confirms that the
    exact checkpoint's licence has been reviewed for the intended use. Do not assume the
    metric Outdoor Small model card is commercially cleared just because the base Small
    model is Apache-2.0.
    """
    DEFAULT_RELATIVE_MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'

    def __init__(self, model_id: str = DEFAULT_RELATIVE_MODEL, *,
                 output_kind: Literal['relative', 'metric'] = 'relative',
                 relative_near_is_high: Optional[bool] = None,
                 device: str = 'auto',
                 metric_license_reviewed: bool = False,
                 revision: str | None = None) -> None:
        if output_kind == 'relative' and relative_near_is_high is None:
            raise ValueError('Set relative_near_is_high explicitly for the chosen checkpoint.')
        if output_kind == 'metric' and not metric_license_reviewed:
            raise ValueError('Metric checkpoint use is blocked until its exact model-weight licence is reviewed.')
        self.model_id = model_id
        self.output_kind = output_kind
        self.relative_near_is_high = relative_near_is_high
        self.device_request = device
        self.metric_license_reviewed = metric_license_reviewed
        self.revision = revision
        self._processor = None
        self._model = None
        self._torch = None
        self._device = None

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError('Install torch and transformers to use this depth adapter.') from exc
        device = resolve_torch_device(torch, self.device_request)
        # Store the runtime before checkpoint loading so close() can clean up even
        # when processor/model initialization fails part-way through.
        self._torch = torch
        self._device = device
        revision_kwargs = {"revision": self.revision} if self.revision else {}
        self._processor = AutoImageProcessor.from_pretrained(self.model_id, **revision_kwargs)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id, **revision_kwargs
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
        """Release model/processor references and clear accelerator allocator cache.

        Safe to call repeatedly. The adapter can be reused; the next inference lazily
        reloads the checkpoint.
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

    def predict_raw(self, rgb_srgb: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return the model's raw prediction before any near/far polarity mapping."""
        if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] != 3:
            raise ValueError('rgb_srgb must have shape HxWx3.')
        if not np.isfinite(rgb_srgb).all() or np.any(rgb_srgb < 0) or np.any(rgb_srgb > 1):
            raise ValueError('rgb_srgb must be finite and in [0,1].')
        image = np.clip(rgb_srgb * 255.0, 0, 255).astype(np.uint8)
        h, w = image.shape[:2]
        if self._model is None:
            self._load()
        from PIL import Image
        torch = self._torch
        pil = Image.fromarray(image, mode='RGB')
        inputs = self._processor(images=pil, return_tensors='pt')
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)
            raw = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False
            )[0, 0].float().cpu().numpy()
        raw = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(raw).all():
            raise RuntimeError('Depth model returned non-finite values.')
        if float(np.max(raw) - np.min(raw)) < 1e-6:
            raise RuntimeError('Depth model returned insufficient dynamic range.')
        return raw, {
            'model_id': self.model_id,
            'device': self._device,
            **revision_descriptor(self.revision),
        }

    def estimate(self, rgb_srgb: np.ndarray) -> DepthPrediction:
        raw, raw_metadata = self.predict_raw(rgb_srgb)
        if self.output_kind == 'metric':
            valid = raw > 0
            if not np.any(valid):
                raise RuntimeError('Metric depth model returned no positive depth values.')
            depth = np.where(valid, raw, 0.0).astype(np.float32)
            kind = 'metric'
        else:
            depth = normalize_relative_depth(raw, near_is_high=bool(self.relative_near_is_high))
            valid = np.ones(raw.shape, dtype=bool)
            kind = 'relative'
        # This model head does not expose calibrated per-pixel uncertainty; use a conservative
        # baseline confidence and label its origin instead of pretending it is model confidence.
        confidence = np.where(valid, 0.45, 0.0).astype(np.float32)
        return DepthPrediction(depth.astype(np.float32), confidence, kind, {
            **raw_metadata,
            'depth_kind': kind,
            'relative_near_is_high': self.relative_near_is_high,
            'depth_units': 'metres' if kind == 'metric' else 'normalized-relative-units',
            'confidence_source': 'conservative constant prior; model does not emit calibrated confidence',
            'metric_license_reviewed_by_caller': self.metric_license_reviewed,
        })
