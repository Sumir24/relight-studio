"""Physics-steered IC-Light V2 (FLUX) adapter supporting Cloud API (fal.ai) and local Diffusers."""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image

from ..color import linear_to_srgb, srgb_to_linear
from ..composite import compose_intrinsic
from ..contracts import EngineSettings, LightingRig, SceneBuffers
from .protocols import NeuralRelightPrediction
from .runtime import (
    memory_snapshot,
    release_device_cache,
    release_model_from_device,
    resolve_torch_device,
)


def _encode_png_data_uri(rgb_srgb: np.ndarray) -> str:
    u8 = np.round(np.clip(rgb_srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8, mode="RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _decode_image_from_url_or_b64(url: str) -> np.ndarray:
    if url.startswith("data:image/"):
        _, b64 = url.split(",", 1)
        raw = base64.b64decode(b64)
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "RelightStudio/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


class FluxICLightRelighter:
    """Tier 2A IC-Light V2 (FLUX) relighter steered by PBR radiance hints.

    Supports both Cloud API execution (`provider="fal"` via `FAL_KEY`, ideal for
    16 GB Apple Silicon laptops) and local Diffusers FLUX inference (`provider="local"`).
    """

    DEFAULT_FAL_ENDPOINT = "https://fal.run/fal-ai/iclight-v2"
    DEFAULT_LOCAL_MODEL = "black-forest-labs/FLUX.1-schnell"

    def __init__(
        self,
        *,
        provider: str = "auto",
        fal_endpoint: str = DEFAULT_FAL_ENDPOINT,
        api_key: str | None = None,
        local_model_id: str = DEFAULT_LOCAL_MODEL,
        device: str = "auto",
        num_inference_steps: int = 8,
        strength: float = 0.55,
        guidance_scale: float = 3.5,
        max_side: int = 1024,
        seed: int = 42,
        prompt: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if provider not in ("auto", "fal", "local"):
            raise ValueError("provider must be 'auto', 'fal', or 'local'.")
        if num_inference_steps < 1 or not 0.0 < strength <= 1.0:
            raise ValueError("num_inference_steps must be >= 1 and strength in (0, 1].")
        if max_side < 64:
            raise ValueError("max_side must be >= 64.")

        self.provider_request = provider
        self.fal_endpoint = fal_endpoint
        self.api_key = api_key
        self.local_model_id = local_model_id
        self.device_request = device
        self.num_inference_steps = int(num_inference_steps)
        self.strength = float(strength)
        self.guidance_scale = float(guidance_scale)
        self.max_side = int(max_side)
        self.seed = int(seed)
        self.prompt_override = prompt
        self.timeout_seconds = float(timeout_seconds)

        self._torch: Any = None
        self._device: str | None = None
        self._pipe: Any = None

    @staticmethod
    def _target_size(width: int, height: int, max_side: int) -> tuple[int, int]:
        scale = min(1.0, float(max_side) / float(max(width, height)))
        snap_w = max(16, ((int(round(width * scale)) + 15) // 16) * 16)
        snap_h = max(16, ((int(round(height * scale)) + 15) // 16) * 16)
        return snap_w, snap_h

    @staticmethod
    def describe_rig_prompt(scene: SceneBuffers, target_rig: LightingRig) -> str:
        """Convert LightingRig geometry & color into an explicit directional relighting prompt."""
        if not target_rig.lights:
            dir_desc = "soft ambient studio illumination"
            color_desc = "natural balanced"
        else:
            primary = max(target_rig.lights, key=lambda l: float(l.strength))
            vec = primary.direction if primary.kind == "directional" else primary.position
            vx, vy, _ = (float(vec[0]), float(vec[1]), float(vec[2])) if vec is not None else (0.0, 0.0, -1.0)
            horiz = "from the left" if vx < -0.2 else ("from the right" if vx > 0.2 else "from front")
            vert = " and above" if vy > 0.35 else (" and below" if vy < -0.35 else "")
            dir_desc = f"{primary.kind} light {horiz}{vert}"

            c = np.asarray(primary.color, dtype=np.float32)
            r, g, b = float(c[0]), float(c[1]), float(c[2])
            if r > max(g, b) * 1.8:
                color_desc = "vivid red"
            elif b > max(r, g) * 1.4:
                color_desc = "cool blue"
            elif g > max(r, b) * 1.4:
                color_desc = "emerald green"
            elif r > b * 1.25:
                color_desc = "warm golden"
            else:
                color_desc = "neutral studio white"

        caption = ""
        if isinstance(scene.metadata, dict):
            disc = scene.metadata.get("scene_discovery", {})
            if isinstance(disc, dict):
                caption = str(disc.get("caption", "")).strip()
        subj = f"{caption}, " if caption else ""
        return (
            f"{subj}photorealistic relighting with {color_desc} {dir_desc}, "
            "natural shadows and specular highlights, preserve original subject identity and surface detail"
        )

    @staticmethod
    def build_radiance_hint_image(
        scene: SceneBuffers,
        target_diffuse: np.ndarray,
        target_specular: np.ndarray,
        target_hw: tuple[int, int],
        settings: EngineSettings,
    ) -> np.ndarray:
        """Render a physics radiance-hint image for neural conditioning (DiLightNet style)."""
        if scene.albedo is not None:
            geom_conf = (
                scene.geometry_confidence
                if scene.geometry_confidence is not None
                else np.ones(scene.shape, dtype=np.float32)
            )
            hint_srgb, _ = compose_intrinsic(
                scene, target_diffuse, target_specular, geom_conf, settings
            )
        else:
            src_lin = srgb_to_linear(scene.rgb_srgb)
            irrad = np.maximum(target_diffuse, 0.0) + np.maximum(target_specular, 0.0) * 0.5
            peak = max(float(np.percentile(irrad, 99.5)), 1.0)
            hint_lin = np.clip(src_lin * (irrad / peak) * 1.4 + np.maximum(target_specular, 0.0) * 0.35, 0.0, 1.0)
            hint_srgb = linear_to_srgb(hint_lin).astype(np.float32)

        th, tw = target_hw
        if hint_srgb.shape[:2] != (th, tw):
            hint_srgb = cv2.resize(hint_srgb, (tw, th), interpolation=cv2.INTER_AREA)
        return np.clip(hint_srgb, 0.0, 1.0).astype(np.float32)

    def _resolve_provider(self) -> str:
        if self.provider_request in ("fal", "local"):
            return self.provider_request
        key = self.api_key or os.environ.get("FAL_KEY")
        if key:
            return "fal"
        return "local"

    def _run_fal(
        self,
        fg_srgb: np.ndarray,
        hint_srgb: np.ndarray,
        prompt: str,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        key = self.api_key or os.environ.get("FAL_KEY")
        if not key:
            raise RuntimeError(
                "FluxICLightRelighter (provider='fal') requires FAL_KEY env var or api_key argument."
            )
        payload = {
            "prompt": prompt,
            "image_url": _encode_png_data_uri(fg_srgb),
            "background_image_url": _encode_png_data_uri(hint_srgb),
            "image_size": {"width": int(target_w), "height": int(target_h)},
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "initial_latent": "None",
            "enable_hr_fix": False,
        }
        req = urllib.request.Request(
            self.fal_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Key {key}",
                "Content-Type": "application/json",
                "User-Agent": "RelightStudio/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"fal.ai IC-Light V2 request failed ({exc.code}): {detail}") from exc

        images = body.get("images") or []
        if not images or not isinstance(images[0], dict) or "url" not in images[0]:
            raise RuntimeError(f"Unexpected fal.ai IC-Light V2 response format: {list(body.keys())}")
        return _decode_image_from_url_or_b64(images[0]["url"])

    def _load_local(self) -> None:
        try:
            import torch
            from diffusers import FluxImg2ImgPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and diffusers>=0.32 with FLUX support for local FluxICLightRelighter, "
                "or set FAL_KEY and use provider='fal'."
            ) from exc

        device = resolve_torch_device(torch, self.device_request)
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
        self._torch = torch
        self._device = device
        pipe = FluxImg2ImgPipeline.from_pretrained(self.local_model_id, torch_dtype=dtype)
        if hasattr(pipe, "enable_model_cpu_offload") and device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device=device, dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    def _run_local(
        self,
        hint_srgb: np.ndarray,
        prompt: str,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        if self._pipe is None:
            self._load_local()
        torch = self._torch
        init_pil = Image.fromarray(np.round(np.clip(hint_srgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
        with torch.inference_mode():
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            out_pil = self._pipe(
                prompt=prompt,
                image=init_pil,
                strength=self.strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                width=target_w,
                height=target_h,
                generator=gen,
            ).images[0]
        return np.asarray(out_pil, dtype=np.float32) / 255.0

    def render(
        self,
        scene: SceneBuffers,
        target_diffuse: np.ndarray,
        target_specular: np.ndarray,
        target_rig: LightingRig,
        settings: EngineSettings,
    ) -> NeuralRelightPrediction:
        h, w = scene.shape
        target_w, target_h = self._target_size(w, h, self.max_side)
        fg_rgb = scene.rgb_srgb
        if fg_rgb.shape[:2] != (target_h, target_w):
            fg_resized = cv2.resize(fg_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            fg_resized = fg_rgb

        hint_srgb = self.build_radiance_hint_image(
            scene, target_diffuse, target_specular, (target_h, target_w), settings
        )
        prompt = self.prompt_override or self.describe_rig_prompt(scene, target_rig)
        active_provider = self._resolve_provider()

        if active_provider == "fal":
            out = self._run_fal(fg_resized, hint_srgb, prompt, target_w, target_h)
        else:
            out = self._run_local(hint_srgb, prompt, target_w, target_h)

        out = np.asarray(out, dtype=np.float32)
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        return NeuralRelightPrediction(
            rgb_srgb=out,
            metadata={
                "backend": "ic_light_v2_flux",
                "provider": active_provider,
                "endpoint_or_model": self.fal_endpoint if active_provider == "fal" else self.local_model_id,
                "device": self._device or active_provider,
                "inference_resolution": [target_w, target_h],
                "prompt": prompt,
                "radiance_hint_used": True,
            },
        )

    def runtime_snapshot(self) -> dict:
        provider = self._resolve_provider()
        return {
            "provider": provider,
            "device": self._device or provider,
            "model_loaded": self._pipe is not None,
            "memory": (
                memory_snapshot(self._torch, self._device)
                if self._torch is not None
                else {"device": self._device or provider, "available": False}
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
