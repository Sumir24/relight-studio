"""Pluggable Diffusers-backed intrinsic image decomposition adapter."""
from __future__ import annotations

from typing import Any, Literal
import cv2
import numpy as np

from ..color import linear_to_srgb, srgb_to_linear
from .protocols import IntrinsicPrediction
from .runtime import (
    memory_snapshot,
    release_device_cache,
    release_model_from_device,
    resolve_torch_device,
    revision_descriptor,
)

IntrinsicVariant = Literal["lighting", "appearance"]


class DiffusersIntrinsicAdapter:
    """Optional Diffusers intrinsic decomposition adapter conforming to IntrinsicEstimator."""

    DEFAULT_LIGHTING_MODEL = "prs-eth/marigold-iid-lighting-v1-1"
    DEFAULT_APPEARANCE_MODEL = "prs-eth/marigold-iid-appearance-v1-1"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        variant_kind: IntrinsicVariant = "lighting",
        device: str = "auto",
        num_inference_steps: int = 4,
        ensemble_size: int = 1,
        processing_resolution: int | None = None,
        output_is_linear: bool = True,
        revision: str | None = None,
    ) -> None:
        if variant_kind not in ("lighting", "appearance"):
            raise ValueError("variant_kind must be 'lighting' or 'appearance'.")
        if num_inference_steps < 1 or ensemble_size < 1:
            raise ValueError("num_inference_steps and ensemble_size must be >= 1.")
        self.variant_kind = variant_kind
        self.model_id = model_id or (
            self.DEFAULT_LIGHTING_MODEL
            if variant_kind == "lighting"
            else self.DEFAULT_APPEARANCE_MODEL
        )
        self.device_request = device
        self.num_inference_steps = int(num_inference_steps)
        self.ensemble_size = int(ensemble_size)
        self.processing_resolution = processing_resolution
        self.output_is_linear = bool(output_is_linear)
        self.revision = revision
        self._torch = None
        self._device: str | None = None
        self._pipe: Any = None

    def _load(self) -> None:
        try:
            import torch
            from diffusers import MarigoldIntrinsicsPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and diffusers>=0.32 to use DiffusersIntrinsicAdapter."
            ) from exc
        device = resolve_torch_device(torch, self.device_request)
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self._torch = torch
        self._device = device
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.revision:
            kwargs["revision"] = self.revision
        pipe = MarigoldIntrinsicsPipeline.from_pretrained(self.model_id, **kwargs)
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

    @staticmethod
    def _align(arr: np.ndarray, h: int, w: int) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float32)
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)

    def estimate(self, rgb_srgb: np.ndarray) -> IntrinsicPrediction:
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
        pred = np.asarray(output.prediction, dtype=np.float32)
        if not np.isfinite(pred).all():
            raise RuntimeError("Intrinsic estimation produced non-finite values.")
        if pred.ndim != 4 or pred.shape[0] < 2 or pred.shape[-1] != 3:
            raise RuntimeError(f"Unexpected intrinsic output shape: {pred.shape}")

        props = getattr(self._pipe, "target_properties", None) or {}
        target_names = [str(n).lower() for n in (props.get("target_names") or [])]
        albedo_idx = target_names.index("albedo") if "albedo" in target_names else 0
        raw_albedo = np.clip(self._align(pred[albedo_idx], h, w), 0.0, 1.0)

        pred_spaces = props.get("prediction_space") or props.get("target_spaces") or {}
        albedo_is_linear = self.output_is_linear
        if isinstance(pred_spaces, dict) and "albedo" in pred_spaces:
            albedo_is_linear = str(pred_spaces["albedo"]).lower() != "srgb"
        albedo = linear_to_srgb(raw_albedo).astype(np.float32) if albedo_is_linear else raw_albedo
        albedo_lin = raw_albedo if albedo_is_linear else srgb_to_linear(raw_albedo)

        residual = None
        roughness = None
        metalness = None
        if self.variant_kind == "lighting":
            sh_idx = target_names.index("shading") if "shading" in target_names else 1
            source_shading = np.maximum(self._align(pred[sh_idx], h, w), 0.0)
            res_idx = target_names.index("residual") if "residual" in target_names else 2
            if res_idx < pred.shape[0]:
                residual = np.maximum(self._align(pred[res_idx], h, w), 0.0)
        elif self.variant_kind == "appearance":
            mat_idx = (
                target_names.index("material")
                if "material" in target_names
                else (1 if albedo_idx == 0 else 0)
            )
            mat = self._align(pred[mat_idx], h, w)
            roughness = np.clip(mat[..., 0], 0.0, 1.0)
            metalness = np.clip(mat[..., 1], 0.0, 1.0)
            rgb_lin = srgb_to_linear(rgb_srgb)
            source_shading = np.maximum(
                rgb_lin / np.maximum(albedo_lin, 1e-3), 0.0
            ).astype(np.float32)
        else:
            source_shading = np.maximum(self._align(pred[1], h, w), 0.0)

        return IntrinsicPrediction(
            albedo=albedo.astype(np.float32),
            source_shading=source_shading.astype(np.float32),
            non_diffuse_residual=residual,
            roughness=roughness,
            metalness=metalness,
            metadata={
                "model_id": self.model_id,
                "variant_kind": self.variant_kind,
                "device": self._device,
                **revision_descriptor(self.revision),
                "source": "model_estimated",
                "calibrated": False,
            },
        )
