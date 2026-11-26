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
        if self.ensemble_size > 2 and self.variant_kind == "appearance":
            call_kwargs["output_uncertainty"] = True
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
        albedo_meta = props.get("albedo") if isinstance(props.get("albedo"), dict) else {}
        albedo_is_linear = self.output_is_linear
        if isinstance(albedo_meta, dict) and "prediction_space" in albedo_meta:
            albedo_is_linear = str(albedo_meta["prediction_space"]).lower() != "srgb"
        elif isinstance(pred_spaces, dict) and "albedo" in pred_spaces:
            albedo_is_linear = str(pred_spaces["albedo"]).lower() != "srgb"
        albedo = linear_to_srgb(raw_albedo).astype(np.float32) if albedo_is_linear else raw_albedo
        albedo_lin = raw_albedo if albedo_is_linear else srgb_to_linear(raw_albedo)

        residual = None
        roughness = None
        metalness = None
        specular_level = None
        specular_color = None
        material_confidence = None
        roughness_unc = None
        metalness_unc = None
        extra_metadata: dict[str, Any] = {
            "model_id": self.model_id,
            "variant_kind": self.variant_kind,
            "device": self._device,
            **revision_descriptor(self.revision),
            "source": "model_estimated",
            "calibrated": False,
        }
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
            raw_roughness = np.clip(mat[..., 0], 0.0, 1.0).astype(np.float32)
            raw_metalness = np.clip(mat[..., 1], 0.0, 1.0).astype(np.float32)
            ch2_unused = mat[..., 2].astype(np.float32) if mat.shape[-1] >= 3 else np.zeros((h, w), np.float32)

            unc = getattr(output, "uncertainty", None)
            if unc is not None:
                unc_arr = np.asarray(unc, dtype=np.float32)
                if unc_arr.ndim == 4 and unc_arr.shape[0] > mat_idx:
                    mat_unc = self._align(unc_arr[mat_idx], h, w)
                    roughness_unc = np.clip(mat_unc[..., 0], 0.0, 1.0).astype(np.float32)
                    metalness_unc = np.clip(mat_unc[..., 1], 0.0, 1.0).astype(np.float32)
                    uncertainty_source = "ensemble_dispersion"
                else:
                    roughness_unc = None
                    metalness_unc = None
                    uncertainty_source = "single_sample_conservative_prior"
            else:
                roughness_unc = None
                metalness_unc = None
                uncertainty_source = "single_sample_conservative_prior"

            # Confidence-aware material assignment:
            # 1. Trust ensemble-agreed predictions (cap 0.86 vs 0.60 uncalibrated single-sample).
            base_cap = 0.86 if roughness_unc is not None else 0.60
            if roughness_unc is not None and metalness_unc is not None:
                unc_combined = 0.65 * roughness_unc + 0.35 * metalness_unc
                unc_factor = np.exp(-unc_combined / 0.065)
            else:
                unc_factor = np.ones((h, w), dtype=np.float32)

            # 2. Gentle shadow-floor SNR damping only near crushed black (<0.02 linear),
            # keeping dark matte textiles from being falsely treated as low-confidence.
            rgb_lin = srgb_to_linear(rgb_srgb)
            lum_lin = (0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]).astype(np.float32)
            snr_factor = np.clip((lum_lin - 0.002) / 0.018, 0.55, 1.0).astype(np.float32)

            # 3. Sanity gate against unexpected spikes in unused channel 2.
            sanity_factor = np.clip(1.0 - np.maximum(np.abs(ch2_unused) - 0.15, 0.0) * 2.0, 0.3, 1.0).astype(np.float32)
            material_confidence = np.clip(
                base_cap * unc_factor * snr_factor * sanity_factor,
                0.0,
                1.0,
            ).astype(np.float32)

            # Keep raw model predictions faithful at the adapter boundary; downstream material-prior
            # assignment separates glossy dielectrics/clearcoat from conductors via uncertainty.
            metalness = raw_metalness
            roughness = raw_roughness
            specular_level = np.full((h, w), 0.04, dtype=np.float32)
            specular_color = np.clip(albedo_lin, 0.04, 0.98).astype(np.float32)
            source_shading = np.maximum(
                rgb_lin / np.maximum(albedo_lin, 1e-3), 0.0
            ).astype(np.float32)
            extra_metadata["material_target_semantics"] = {
                "sub_target_names": ["roughness", "metallicity", None],
                "uncertainty_source": uncertainty_source,
                "license": "openrail++",
                "raw_roughness_mean": float(np.mean(raw_roughness)),
                "raw_metalness_mean": float(np.mean(raw_metalness)),
                "material_confidence_mean": float(np.mean(material_confidence)),
            }
            if roughness_unc is not None and metalness_unc is not None:
                extra_metadata["material_target_semantics"]["raw_roughness_unc_mean"] = float(np.mean(roughness_unc))
                extra_metadata["material_target_semantics"]["raw_metalness_unc_mean"] = float(np.mean(metalness_unc))
        else:
            source_shading = np.maximum(self._align(pred[1], h, w), 0.0)

        return IntrinsicPrediction(
            albedo=albedo.astype(np.float32),
            source_shading=source_shading.astype(np.float32),
            non_diffuse_residual=residual,
            roughness=roughness,
            metalness=metalness,
            specular_level=specular_level,
            specular_color=specular_color,
            material_confidence=material_confidence,
            roughness_uncertainty=roughness_unc,
            metalness_uncertainty=metalness_unc,
            metadata=extra_metadata,
        )
