"""Pluggable Diffusers-backed surface normal adapter (NormalEstimator protocol)."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .protocols import NormalPrediction
from .runtime import (
    memory_snapshot,
    release_device_cache,
    release_model_from_device,
    resolve_torch_device,
    revision_descriptor,
)


class DiffusersNormalAdapter:
    """Optional Diffusers surface-normal estimator conforming to NormalEstimator."""

    DEFAULT_MODEL = "prs-eth/marigold-normals-v1-1"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        num_inference_steps: int = 4,
        ensemble_size: int = 1,
        processing_resolution: int | None = None,
        camera_z_toward: bool = True,
        revision: str | None = None,
    ) -> None:
        if num_inference_steps < 1 or ensemble_size < 1:
            raise ValueError("num_inference_steps and ensemble_size must be >= 1.")
        self.model_id = model_id
        self.device_request = device
        self.num_inference_steps = int(num_inference_steps)
        self.ensemble_size = int(ensemble_size)
        self.processing_resolution = processing_resolution
        self.camera_z_toward = bool(camera_z_toward)
        self.revision = revision
        self._torch = None
        self._device: str | None = None
        self._pipe: Any = None

    def _load(self) -> None:
        try:
            import torch
            from diffusers import MarigoldNormalsPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and diffusers>=0.32 to use DiffusersNormalAdapter."
            ) from exc
        device = resolve_torch_device(torch, self.device_request)
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self._torch = torch
        self._device = device
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.revision:
            kwargs["revision"] = self.revision
        pipe = MarigoldNormalsPipeline.from_pretrained(self.model_id, **kwargs)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe.to(device)

    def runtime_snapshot(self) -> dict:
        return {
            "device": self._device,
            "model_loaded": self._pipe is not None,
            "memory": (
                memory_snapshot(self._torch, self._device)
                if self._torch is not None
                else {"device": self._device, "available": False}
            ),
        }

    def close(self) -> dict:
        if self._torch is None:
            return {"released": False, "already_closed": True, "device": self._device}
        before = memory_snapshot(self._torch, self._device)
        was_loaded = self._pipe is not None
        release_model_from_device(self._pipe)
        self._pipe = None
        cleanup = release_device_cache(self._torch, self._device)
        after = memory_snapshot(self._torch, self._device)
        return {
            "released": was_loaded,
            "device": self._device,
            "memory_before_release": before,
            "memory_after_release": after,
            "cleanup": cleanup,
        }

    def estimate(
        self, rgb_srgb: np.ndarray, *, depth: np.ndarray | None = None
    ) -> NormalPrediction:
        del depth  # Adapter estimates normals directly from RGB.
        if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] != 3:
            raise ValueError("rgb_srgb must have shape HxWx3.")
        if not np.isfinite(rgb_srgb).all() or np.any(rgb_srgb < 0) or np.any(rgb_srgb > 1):
            raise ValueError("rgb_srgb must be finite and in [0,1].")
        h, w = rgb_srgb.shape[:2]
        if self._pipe is None:
            self._load()
        from PIL import Image

        pil = Image.fromarray(np.round(rgb_srgb * 255.0).astype(np.uint8), mode="RGB")
        call_kwargs: dict[str, Any] = {
            "num_inference_steps": self.num_inference_steps,
            "ensemble_size": self.ensemble_size,
            "match_input_resolution": True,
            "output_type": "np",
        }
        if self.processing_resolution is not None:
            call_kwargs["processing_resolution"] = int(self.processing_resolution)
        with self._torch.inference_mode():
            output = self._pipe(pil, **call_kwargs)
        raw = np.asarray(output.prediction, dtype=np.float32)
        if not np.isfinite(raw).all():
            raise RuntimeError("Surface normal estimation produced non-finite values.")
        if raw.ndim == 4:
            raw = raw[0]
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise RuntimeError(f"Unexpected normal output shape: {raw.shape}")
        if raw.shape[:2] != (h, w):
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.camera_z_toward:
            raw = raw.copy()
            raw[..., 2] = -raw[..., 2]
        norms = np.linalg.norm(raw, axis=-1, keepdims=True)
        valid = norms[..., 0] > 1e-6
        normals = raw / np.maximum(norms, 1e-6)
        normals[~valid] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        confidence = np.where(valid, 0.75, 0.0).astype(np.float32)
        return NormalPrediction(
            normals=normals.astype(np.float32),
            confidence=confidence,
            metadata={
                "model_id": self.model_id,
                "device": self._device,
                **revision_descriptor(self.revision),
                "confidence_source": "conservative prior on unit-normal validity; not calibrated uncertainty",
            },
        )
