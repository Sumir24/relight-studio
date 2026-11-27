"""Physics-steered IC-Light SD1.5 adapter conforming to NeuralRelighter."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from ..color import linear_to_srgb, srgb_to_linear
from ..composite import compose_intrinsic
from ..contracts import EngineSettings, LightingRig, SceneBuffers
from .protocols import NeuralRelightPrediction
from .runtime import (
    memory_snapshot,
    release_device_cache,
    release_model_from_device,
    resolve_torch_device,
    revision_descriptor,
)


class ICLightRelighter:
    """Optional SD1.5 IC-Light relighter steered by physics-rendered target shading."""

    DEFAULT_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    DEFAULT_ICLIGHT_REPO = "lllyasviel/ic-light"
    DEFAULT_WEIGHTS_FILE = "iclight_sd15_fc.safetensors"

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL,
        *,
        iclight_repo_id: str = DEFAULT_ICLIGHT_REPO,
        weights_filename: str = DEFAULT_WEIGHTS_FILE,
        device: str = "auto",
        num_inference_steps: int = 20,
        strength: float = 0.58,
        guidance_scale: float = 2.0,
        max_side: int = 768,
        seed: int = 42,
        prompt: str | None = None,
        base_revision: str | None = None,
        revision: str | None = None,
    ) -> None:
        if num_inference_steps < 1 or not 0.0 < strength <= 1.0:
            raise ValueError("num_inference_steps must be >= 1 and strength in (0, 1].")
        if max_side < 64:
            raise ValueError("max_side must be >= 64.")
        self.base_model_id = base_model_id
        self.iclight_repo_id = iclight_repo_id
        self.weights_filename = weights_filename
        self.device_request = device
        self.num_inference_steps = int(num_inference_steps)
        self.strength = float(strength)
        self.guidance_scale = float(guidance_scale)
        self.max_side = int(max_side)
        self.seed = int(seed)
        self.prompt_override = prompt
        self.base_revision = base_revision
        self.revision = revision
        self._torch: Any = None
        self._device: str | None = None
        self._pipe: Any = None
        self._concat_conds: Any = None

    @staticmethod
    def _target_size(width: int, height: int, max_side: int) -> tuple[int, int]:
        scale = min(1.0, float(max_side) / float(max(width, height)))
        scaled_w = max(8, int(round(width * scale)))
        scaled_h = max(8, int(round(height * scale)))
        snap_w = ((scaled_w + 7) // 8) * 8
        snap_h = ((scaled_h + 7) // 8) * 8
        return snap_w, snap_h

    @staticmethod
    def build_physics_lighting_image(
        target_diffuse: np.ndarray,
        target_specular: np.ndarray,
        target_hw: tuple[int, int],
    ) -> np.ndarray:
        """Convert linear target shading into an sRGB [0,1] conditioning image."""
        combined = np.maximum(target_diffuse, 0.0) + np.maximum(target_specular, 0.0) * 0.5
        peak = float(np.percentile(combined, 99.5))
        norm = combined / max(peak, 1.0)
        srgb = linear_to_srgb(np.clip(norm, 0.0, 1.0)).astype(np.float32)
        th, tw = target_hw
        if srgb.shape[:2] != (th, tw):
            srgb = cv2.resize(srgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        return np.clip(srgb, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _describe_rig_prompt(scene: SceneBuffers, target_rig: LightingRig) -> str:
        warmth = "natural daylight"
        if target_rig.lights:
            c = np.asarray(target_rig.lights[0].color, dtype=np.float32)
            if float(c[0]) > float(c[2]) * 1.25:
                warmth = "warm golden sunset directional sunlight"
            elif float(c[2]) > float(c[0]) * 1.2:
                warmth = "cool directional sky lighting"
            else:
                warmth = "balanced directional studio sunlight"
        caption = ""
        if isinstance(scene.metadata, dict):
            disc = scene.metadata.get("scene_discovery", {})
            if isinstance(disc, dict):
                caption = str(disc.get("caption", "")).strip()
        base = f", {caption}" if caption else ""
        return f"{warmth}, realistic cast shadows, soft bounce light, detailed photograph{base}"

    def _load(self) -> None:
        try:
            import torch
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, diffusers>=0.32, huggingface_hub, and safetensors to use ICLightRelighter."
            ) from exc

        device = resolve_torch_device(torch, self.device_request)
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self._torch = torch
        self._device = device

        pipe_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        if self.base_revision:
            pipe_kwargs["revision"] = self.base_revision
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.base_model_id, **pipe_kwargs)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        pipe.set_progress_bar_config(disable=True)

        dl_kwargs: dict[str, Any] = {}
        if self.revision:
            dl_kwargs["revision"] = self.revision
        weights_path = hf_hub_download(
            repo_id=self.iclight_repo_id, filename=self.weights_filename, **dl_kwargs
        )
        sd_offset = load_file(weights_path)

        self._patch_unet_for_iclight(torch, pipe.unet, sd_offset)
        unet = pipe.unet

        original_forward = unet.forward
        adapter_ref = self

        def hooked_unet_forward(sample, timestep, encoder_hidden_states, *args, **kwargs):
            conds = adapter_ref._concat_conds
            if conds is None:
                raise RuntimeError("ICLightRelighter concat conditioning tensor was not prepared.")
            c = conds.to(device=sample.device, dtype=sample.dtype)
            if c.shape[0] != sample.shape[0]:
                c = c.repeat(sample.shape[0] // c.shape[0], 1, 1, 1)
            combined = torch.cat([sample, c], dim=1)
            return original_forward(combined, timestep, encoder_hidden_states, *args, **kwargs)

        unet.forward = hooked_unet_forward
        self._pipe = pipe.to(device=device, dtype=dtype)

    @staticmethod
    def _patch_unet_for_iclight(torch: Any, unet: Any, sd_offset: dict[str, Any]) -> None:
        conv_dtype = unet.conv_in.weight.dtype
        conv_device = unet.conv_in.weight.device
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                8,
                unet.conv_in.out_channels,
                unet.conv_in.kernel_size,
                unet.conv_in.stride,
                unet.conv_in.padding,
                bias=(unet.conv_in.bias is not None),
                device=conv_device,
                dtype=conv_dtype,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            if unet.conv_in.bias is not None and new_conv_in.bias is not None:
                new_conv_in.bias.copy_(unet.conv_in.bias)
            unet.conv_in = new_conv_in
            unet.config["in_channels"] = 8

        unet_state = unet.state_dict()
        matched = 0
        for key, offset in sd_offset.items():
            if key in unet_state:
                unet_state[key] = unet_state[key] + offset.to(
                    dtype=unet_state[key].dtype, device=unet_state[key].device
                )
                matched += 1
        if matched == 0:
            raise RuntimeError("IC-Light offset checkpoint matched zero keys in the base UNet.")
        unet.load_state_dict(unet_state)

    def _run_diffusion(
        self,
        fg_u8: np.ndarray,
        bg_u8: np.ndarray,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
    ) -> np.ndarray:
        from PIL import Image
        del width, height

        torch = self._torch
        fg_tensor = (
            torch.from_numpy(fg_u8.astype(np.float32) / 127.5 - 1.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self._device, dtype=self._pipe.vae.dtype)
        )
        try:
            with torch.inference_mode():
                latents = self._pipe.vae.encode(fg_tensor).latent_dist.mode()
                self._concat_conds = latents * self._pipe.vae.config.scaling_factor
                generator = torch.Generator(device="cpu").manual_seed(self.seed)
                bg_pil = Image.fromarray(bg_u8, mode="RGB")
                result = self._pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=bg_pil,
                    strength=self.strength,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                ).images[0]
        finally:
            self._concat_conds = None
        return np.asarray(result, dtype=np.float32) / 255.0

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
        self._concat_conds = None
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

    @classmethod
    def build_physics_init_image(
        cls,
        scene: SceneBuffers,
        target_diffuse: np.ndarray,
        target_specular: np.ndarray,
        target_hw: tuple[int, int],
        settings: EngineSettings,
    ) -> np.ndarray:
        """Build a scene-colored physics-relighted Img2Img seed so IC-Light preserves scene hues."""
        if scene.albedo is not None:
            geom_conf = (
                scene.geometry_confidence
                if scene.geometry_confidence is not None
                else np.ones(scene.shape, dtype=np.float32)
            )
            guide_srgb, _ = compose_intrinsic(
                scene, target_diffuse, target_specular, geom_conf, settings
            )
        else:
            src_lin = srgb_to_linear(scene.rgb_srgb)
            irrad = np.maximum(target_diffuse, 0.0) + np.maximum(target_specular, 0.0) * 0.5
            peak = max(float(np.percentile(irrad, 99.5)), 1.0)
            shaded_lin = np.clip(src_lin * (irrad / peak) * 1.6 + np.maximum(target_specular, 0.0) * 0.35, 0.0, 1.0)
            guide_srgb = linear_to_srgb(shaded_lin).astype(np.float32)
        th, tw = target_hw
        if guide_srgb.shape[:2] != (th, tw):
            guide_srgb = cv2.resize(guide_srgb, (tw, th), interpolation=cv2.INTER_AREA)
        return np.clip(guide_srgb, 0.0, 1.0).astype(np.float32)

    def render(
        self,
        scene: SceneBuffers,
        target_diffuse: np.ndarray,
        target_specular: np.ndarray,
        target_rig: LightingRig,
        settings: EngineSettings,
    ) -> NeuralRelightPrediction:
        if self._pipe is None:
            self._load()
        h, w = scene.shape
        target_w, target_h = self._target_size(w, h, self.max_side)

        # Prefer original RGB as structural fg guide; intrinsic albedo can wash out unlit contrast.
        fg_rgb = scene.rgb_srgb
        if fg_rgb.shape[:2] != (target_h, target_w):
            fg_resized = cv2.resize(fg_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            fg_resized = fg_rgb
        fg_u8 = np.round(np.clip(fg_resized, 0.0, 1.0) * 255.0).astype(np.uint8)

        bg_cond = self.build_physics_init_image(
            scene, target_diffuse, target_specular, (target_h, target_w), settings
        )
        bg_u8 = np.round(np.clip(bg_cond, 0.0, 1.0) * 255.0).astype(np.uint8)

        prompt = self.prompt_override or self._describe_rig_prompt(scene, target_rig)
        negative_prompt = "blurry, distorted, cartoon, illustration, overexposed, lowres"

        out = self._run_diffusion(
            fg_u8, bg_u8, prompt, negative_prompt, target_w, target_h
        )
        out = np.asarray(out, dtype=np.float32)
        if out.shape[:2] != (h, w):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        return NeuralRelightPrediction(
            rgb_srgb=out,
            metadata={
                "backend": "ic_light",
                "base_model_id": self.base_model_id,
                "iclight_repo_id": self.iclight_repo_id,
                "weights_filename": self.weights_filename,
                "device": self._device,
                "inference_resolution": [target_w, target_h],
                "prompt": prompt,
                "base_revision": revision_descriptor(self.base_revision),
                "revision": revision_descriptor(self.revision),
            },
        )
