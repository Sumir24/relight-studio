from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest

from relight_engine.adapters.protocols import IntrinsicEstimator, NormalEstimator
from relight_engine.adapters.registry import (
    create_intrinsic_backend,
    create_normal_backend,
)


def test_diffusers_normal_and_intrinsic_adapters_resize_validate_and_close_cleanly(monkeypatch):
    from relight_engine.adapters.diffusers_normals import DiffusersNormalAdapter
    from relight_engine.adapters.diffusers_intrinsics import DiffusersIntrinsicAdapter

    seen: dict = {"calls": []}

    class FakePipe:
        def __init__(self, kind: str):
            self.kind = kind
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)
            return self

        def set_progress_bar_config(self, **_kwargs):
            return None

        def __call__(self, image, **_kwargs):
            seen["calls"].append((self.kind, image.size))
            h_out, w_out = 10, 12
            if self.kind == "normals":
                # Output unnormalized normals pointing toward camera (+Z in external convention)
                pred = np.zeros((1, h_out, w_out, 3), dtype=np.float32)
                pred[..., 2] = 2.0
                return SimpleNamespace(prediction=pred)
            # MarigoldIntrinsicsPipeline lighting returns linear [albedo, shading, residual]
            pred = np.stack([
                np.full((h_out, w_out, 3), 0.25, dtype=np.float32),
                np.full((h_out, w_out, 3), 0.8, dtype=np.float32),
                np.full((h_out, w_out, 3), 0.05, dtype=np.float32),
            ], axis=0)
            return SimpleNamespace(prediction=pred, target_properties=["albedo", "shading", "residual"])

    class FakeNormalsPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            seen["normals_load"] = (model_id, kwargs)
            return FakePipe("normals")

    class FakeIntrinsicsPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            seen["intrinsics_load"] = (model_id, kwargs)
            return FakePipe("intrinsics")

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.MarigoldNormalsPipeline = FakeNormalsPipeline
    fake_diffusers.MarigoldIntrinsicsPipeline = FakeIntrinsicsPipeline

    class FakeMPS:
        def __init__(self):
            self.syncs = 0
            self.empties = 0

        def current_allocated_memory(self):
            return 0

        def driver_allocated_memory(self):
            return 0

        def recommended_max_memory(self):
            return 1024

        def synchronize(self):
            self.syncs += 1

        def empty_cache(self):
            self.empties += 1

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.mps = FakeMPS()
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    from contextlib import nullcontext
    fake_torch.inference_mode = lambda: nullcontext()

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    rev = "a" * 40
    normal_adapter = DiffusersNormalAdapter(device="mps", revision=rev)
    intrinsic_adapter = DiffusersIntrinsicAdapter(device="mps", revision=rev)
    assert isinstance(normal_adapter, NormalEstimator)
    assert isinstance(intrinsic_adapter, IntrinsicEstimator)

    rgb = np.full((15, 21, 3), 0.4, dtype=np.float32)
    norm_pred = normal_adapter.estimate(rgb)
    assert norm_pred.normals.shape == (15, 21, 3)
    assert np.allclose(np.linalg.norm(norm_pred.normals, axis=-1), 1.0, atol=1e-5)
    # Positive +Z camera-toward prediction is flipped unconditionally to Relight Studio's -Z camera-facing convention
    assert float(np.mean(norm_pred.normals[..., 2])) == pytest.approx(-1.0, abs=1e-5)
    assert norm_pred.confidence.shape == (15, 21)
    assert seen["normals_load"][1]["revision"] == rev
    assert normal_adapter.close()["released"] is True
    assert normal_adapter.close()["released"] is False

    intr_pred = intrinsic_adapter.estimate(rgb)
    assert intr_pred.albedo.shape == (15, 21, 3)
    # Diffusers Marigold IID predicts linear 0.25 -> converted to ~0.537 sRGB
    assert float(np.mean(intr_pred.albedo)) > 0.45
    assert intr_pred.source_shading.shape == (15, 21, 3)
    assert intr_pred.non_diffuse_residual is not None
    assert intr_pred.non_diffuse_residual.shape == (15, 21, 3)
    assert seen["intrinsics_load"][1]["revision"] == rev
    assert intrinsic_adapter.close()["released"] is True
    assert fake_torch.mps.syncs >= 2
    assert fake_torch.mps.empties >= 2

    # Registry exposes them without hardcoding in callers
    assert isinstance(create_normal_backend("marigold_normals", device="cpu"), NormalEstimator)
    assert isinstance(create_intrinsic_backend("marigold_intrinsics_lighting", device="cpu"), IntrinsicEstimator)


def test_ic_light_relighter_builds_physics_shading_map_and_preserves_resolution(monkeypatch):
    from contextlib import nullcontext
    from relight_engine.adapters.ic_light_relighter import ICLightRelighter
    from relight_engine.adapters.protocols import NeuralRelighter
    from relight_engine.adapters.registry import create_relighter_backend
    from relight_engine.contracts import EngineSettings, Light, LightingRig, SceneBuffers

    assert isinstance(create_relighter_backend("ic_light", device="cpu"), NeuralRelighter)

    h, w = 13, 19  # Odd dimensions to test padding/resizing to multiple of 8 and exact restore
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[..., 2] = -1.0
    scene = SceneBuffers(
        rgb_srgb=np.full((h, w, 3), 0.45, dtype=np.float32),
        depth=np.full((h, w), 2.0, dtype=np.float32),
        depth_confidence=np.ones((h, w), dtype=np.float32),
        lightable_mask=np.ones((h, w), dtype=np.float32),
        depth_kind="relative",
        normals=normals,
        geometry_confidence=np.ones((h, w), dtype=np.float32),
    )
    target_d = np.broadcast_to(np.array([0.85, 0.55, 0.25], dtype=np.float32), (h, w, 3)).copy()
    target_s = np.full((h, w, 3), 0.05, dtype=np.float32)
    rig = LightingRig(ambient_strength=0.15, lights=(
        Light("warm_key", "directional", direction=(-0.7, 0.4, -0.5), color=(1.0, 0.7, 0.4)),
    ))

    conditioning = ICLightRelighter.build_physics_lighting_image(target_d, target_s, (16, 24))
    assert conditioning.shape == (16, 24, 3)
    assert conditioning.dtype == np.float32
    assert float(conditioning.min()) >= 0.0 and float(conditioning.max()) <= 1.0
    # Warm color cast preserved from target_diffuse
    assert float(np.mean(conditioning[..., 0])) > float(np.mean(conditioning[..., 2]))

    # Verify adapter render() restores exact odd (H, W) and close() releases device
    relighter = ICLightRelighter(device="cpu", num_inference_steps=2)
    seen = {}

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"

        @staticmethod
        def inference_mode():
            return nullcontext()

    def fake_run_diffusion(fg_u8, bg_u8, prompt, negative_prompt, width, height):
        seen["size"] = (width, height)
        seen["prompt"] = prompt
        seen["fg_shape"] = fg_u8.shape
        seen["bg_shape"] = bg_u8.shape
        seen["bg_u8"] = bg_u8.copy()
        return np.full((height, width, 3), 0.65, dtype=np.float32)

    relighter._torch = FakeTorch()
    relighter._device = "cpu"
    relighter._pipe = object()
    relighter._run_diffusion = fake_run_diffusion

    # Paint a deep-blue lower-half ocean on scene.rgb_srgb; bg_u8 init guide must preserve scene chroma (B > R in ocean)
    blue_scene = SceneBuffers(
        rgb_srgb=np.broadcast_to(np.array([0.12, 0.32, 0.55], dtype=np.float32), (h, w, 3)).copy(),
        depth=scene.depth,
        depth_confidence=scene.depth_confidence,
        lightable_mask=scene.lightable_mask,
        depth_kind="relative",
        normals=normals,
        geometry_confidence=scene.geometry_confidence,
    )
    pred = relighter.render(blue_scene, target_d, target_s, rig, EngineSettings())
    assert pred.rgb_srgb.shape == (h, w, 3)
    assert seen["size"][0] % 8 == 0 and seen["size"][1] % 8 == 0
    assert seen["size"][0] >= w and seen["size"][1] >= h
    assert float(np.mean(seen["bg_u8"][..., 2])) > float(np.mean(seen["bg_u8"][..., 0]))
    assert relighter.close()["released"] is True
    assert relighter.close()["released"] is False


def test_ic_light_unet_patch_preserves_half_precision_dtype():
    torch = pytest.importorskip("torch")
    from relight_engine.adapters.ic_light_relighter import ICLightRelighter

    class TinyUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = torch.nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=True).to(dtype=torch.float16)
            self.config = {"in_channels": 4}

        def forward(self, sample):
            return self.conv_in(sample)

    unet = TinyUNet()
    offset = {
        "conv_in.weight": torch.ones((8, 8, 3, 3), dtype=torch.float32) * 0.25,
        "conv_in.bias": torch.ones((8,), dtype=torch.float32) * 0.5,
    }
    ICLightRelighter._patch_unet_for_iclight(torch, unet, offset)
    assert unet.conv_in.in_channels == 8
    assert unet.conv_in.weight.dtype == torch.float16
    assert unet.conv_in.bias is not None and unet.conv_in.bias.dtype == torch.float16
    out = unet(torch.zeros((1, 8, 4, 4), dtype=torch.float16))
    assert out.shape == (1, 8, 4, 4)
    assert out.dtype == torch.float16

    # Zero matching keys must raise RuntimeError rather than silently leaving zero weights
    fresh_unet = TinyUNet()
    with pytest.raises(RuntimeError, match="matched zero"):
        ICLightRelighter._patch_unet_for_iclight(torch, fresh_unet, {"unrelated.weight": torch.zeros(1)})


def test_diffusers_intrinsic_appearance_computes_linear_shading_and_rejects_nonfinite(monkeypatch):
    from contextlib import nullcontext
    from types import ModuleType, SimpleNamespace
    import sys
    from relight_engine.adapters.diffusers_intrinsics import DiffusersIntrinsicAdapter
    from relight_engine.adapters.diffusers_normals import DiffusersNormalAdapter
    from relight_engine.color import srgb_to_linear

    fake_diffusers = ModuleType("diffusers")

    class _AppearancePipe:
        target_properties = {"target_names": ["albedo", "material"]}

        @classmethod
        def from_pretrained(cls, *_a, **_kw):
            return cls()

        def set_progress_bar_config(self, **_kw):
            return None

        def to(self, _d):
            return self

        def __call__(self, *_a, **_kw):
            # [albedo_linear=0.2, material=(roughness=0.35, metalness=0.1, unused)]
            albedo = np.full((8, 10, 3), 0.20, dtype=np.float32)
            mat = np.zeros((8, 10, 3), dtype=np.float32)
            mat[..., 0] = 0.35
            mat[..., 1] = 0.10
            return SimpleNamespace(prediction=np.stack([albedo, mat], axis=0))

    class _NaNPipe:
        @classmethod
        def from_pretrained(cls, *_a, **_kw):
            return cls()

        def set_progress_bar_config(self, **_kw):
            return None

        def to(self, _d):
            return self

        def __call__(self, *_a, **_kw):
            bad = np.full((1, 8, 10, 3), np.nan, dtype=np.float32)
            return SimpleNamespace(prediction=bad)

    fake_diffusers.MarigoldIntrinsicsPipeline = _AppearancePipe
    fake_diffusers.MarigoldNormalsPipeline = _NaNPipe
    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_torch.inference_mode = lambda: nullcontext()
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    rgb = np.full((8, 10, 3), 0.50, dtype=np.float32)
    adapter = DiffusersIntrinsicAdapter(variant_kind="appearance", device="cpu")
    pred = adapter.estimate(rgb)
    # Shading in appearance mode must divide in linear RGB space: srgb_to_linear(0.5) / 0.2
    expected_shading = float(srgb_to_linear(np.array([0.50], dtype=np.float32))[0] / 0.20)
    assert float(np.mean(pred.source_shading)) == pytest.approx(expected_shading, rel=1e-3)
    assert pred.roughness is not None and float(np.mean(pred.roughness)) == pytest.approx(0.35, abs=1e-5)
    assert pred.metalness is not None and float(np.mean(pred.metalness)) == pytest.approx(0.10, abs=1e-5)

    norm_adapter = DiffusersNormalAdapter(device="cpu")
    with pytest.raises(RuntimeError, match="non-finite"):
        norm_adapter.estimate(rgb)


def test_ic_light_prompt_puts_lighting_first_and_resets_concat_conds_on_error():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from relight_engine.adapters.ic_light_relighter import ICLightRelighter
    from relight_engine.contracts import Light, LightingRig, SceneBuffers

    scene = SceneBuffers(
        rgb_srgb=np.full((8, 8, 3), 0.5, dtype=np.float32),
        depth=np.full((8, 8), 2.0, dtype=np.float32),
        depth_confidence=np.ones((8, 8), dtype=np.float32),
        lightable_mask=np.ones((8, 8), dtype=np.float32),
        depth_kind="relative",
        metadata={"scene_discovery": {"caption": "a very long caption about architecture and ocean"}},
    )
    rig = LightingRig(ambient_strength=0.1, lights=(
        Light("warm", "directional", direction=(-0.5, 0.5, -0.5), color=(1.0, 0.65, 0.35)),
    ))
    prompt = ICLightRelighter._describe_rig_prompt(scene, rig)
    # Lighting directive must appear before caption so CLIP 77-token truncation never loses light direction/color
    assert prompt.index("warm") < prompt.index("architecture")

    relighter = ICLightRelighter(device="cpu")
    relighter._device = "cpu"
    relighter._torch = SimpleNamespace(
        inference_mode=lambda: nullcontext(),
        Generator=lambda **_kw: SimpleNamespace(manual_seed=lambda _s: None),
        from_numpy=lambda arr: SimpleNamespace(
            permute=lambda *_a: SimpleNamespace(unsqueeze=lambda *_b: SimpleNamespace(to=lambda **_c: arr))
        ),
    )

    def _boom(**_kw):
        raise RuntimeError("diffusion exploded")

    relighter._pipe = SimpleNamespace(
        vae=SimpleNamespace(
            dtype="float32",
            config=SimpleNamespace(scaling_factor=0.18215),
            encode=lambda _x: SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: 1.0)),
        ),
        __call__=_boom,
    )
    # Wrap as callable object so pipe(**kw) invokes _boom
    class _PipeObj:
        vae = relighter._pipe.vae
        def __call__(self, **_kw):
            raise RuntimeError("diffusion exploded")

    relighter._pipe = _PipeObj()
    u8 = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="diffusion exploded"):
        relighter._run_diffusion(u8, u8, "p", "n", 8, 8)
    assert relighter._concat_conds is None
