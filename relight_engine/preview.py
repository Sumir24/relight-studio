from __future__ import annotations

from copy import copy
import cv2
import numpy as np

from .contracts import CameraModel, SceneBuffers


def _resize_map(arr: np.ndarray, target_wh: tuple[int, int], *, is_mask: bool = False) -> np.ndarray:
    tw, th = target_wh
    interp = cv2.INTER_LINEAR if is_mask else cv2.INTER_AREA
    out = cv2.resize(arr.astype(np.float32), (tw, th), interpolation=interp)
    if arr.ndim == 2 and out.ndim == 3:
        out = out[..., 0]
    return out.astype(np.float32)


def downsample_scene_and_arrays_for_preview(
    scene: SceneBuffers,
    camera: CameraModel,
    max_side: int,
    *,
    arrays: dict[str, np.ndarray] | None = None,
) -> tuple[SceneBuffers, CameraModel, dict[str, np.ndarray]]:
    """Canonical preview downsampler for scene buffers, camera intrinsics, and optional NPZ mask dicts."""
    h, w = scene.shape
    longest = max(h, w)
    if longest <= max_side:
        copied_arrays = {k: np.array(v, copy=True) for k, v in arrays.items()} if arrays is not None else {}
        return copy(scene), camera, copied_arrays

    scale = float(max_side) / float(longest)
    tw = max(4, int(round(w * scale)))
    th = max(4, int(round(h * scale)))
    sx = float(tw) / float(w)
    sy = float(th) / float(h)

    scaled_camera = CameraModel(
        fx=camera.fx * sx,
        fy=camera.fy * sy,
        cx=camera.cx * sx,
        cy=camera.cy * sy,
    )
    scaled_normals = None
    if scene.normals is not None:
        nrm = _resize_map(scene.normals, (tw, th))
        lens = np.linalg.norm(nrm, axis=-1, keepdims=True)
        valid = lens[..., 0] > 1e-6
        nrm = np.where(
            valid[..., None],
            nrm / np.maximum(lens, 1e-6),
            np.array([0.0, 0.0, -1.0], np.float32),
        )
        scaled_normals = nrm.astype(np.float32)

    def _opt(arr: np.ndarray | None, *, mask: bool = False, clip01: bool = False) -> np.ndarray | None:
        if arr is None:
            return None
        resized = _resize_map(arr, (tw, th), is_mask=mask)
        return np.clip(resized, 0.0, 1.0).astype(np.float32) if clip01 else resized

    scaled_arrays: dict[str, np.ndarray] = {}
    if arrays is not None:
        for key, val in arrays.items():
            if key.startswith("mask_") or key in ("lightable_mask", "protected_mask", "alpha_mask", "depth_confidence", "geometry_confidence", "material_confidence"):
                scaled_arrays[key] = np.clip(_resize_map(val, (tw, th), is_mask=True), 0.0, 1.0)
            elif key == "normals" and scaled_normals is not None:
                scaled_arrays[key] = scaled_normals
            elif key in ("depth", "source_shading", "non_diffuse_residual"):
                scaled_arrays[key] = np.maximum(_resize_map(val, (tw, th)), 0.0)
            elif key == "anisotropy_angle":
                scaled_arrays[key] = _resize_map(val, (tw, th))
            else:
                scaled_arrays[key] = np.clip(_resize_map(val, (tw, th)), 0.0, 1.0)

    scaled_scene = SceneBuffers(
        rgb_srgb=np.clip(_resize_map(scene.rgb_srgb, (tw, th)), 0.0, 1.0),
        depth=np.maximum(_resize_map(scene.depth, (tw, th)), 0.0),
        depth_confidence=np.clip(_resize_map(scene.depth_confidence, (tw, th), is_mask=True), 0.0, 1.0),
        lightable_mask=np.clip(_resize_map(scene.lightable_mask, (tw, th), is_mask=True), 0.0, 1.0),
        depth_kind=scene.depth_kind,
        roughness=_opt(scene.roughness, clip01=True),
        metalness=_opt(scene.metalness, clip01=True),
        specular_level=_opt(scene.specular_level, clip01=True),
        clearcoat=_opt(scene.clearcoat, clip01=True),
        clearcoat_roughness=_opt(scene.clearcoat_roughness, clip01=True),
        subsurface=_opt(scene.subsurface, clip01=True),
        anisotropy=_opt(scene.anisotropy, clip01=True),
        anisotropy_angle=_opt(scene.anisotropy_angle),
        normals=scaled_normals,
        geometry_confidence=_opt(scene.geometry_confidence, mask=True, clip01=True),
        alpha_mask=_opt(scene.alpha_mask, mask=True, clip01=True),
        metadata=dict(scene.metadata),
        protected_mask=_opt(scene.protected_mask, mask=True, clip01=True),
        semantic_masks={
            k: np.clip(_resize_map(v, (tw, th), is_mask=True), 0.0, 1.0)
            for k, v in scene.semantic_masks.items()
        },
        instance_masks={
            k: np.clip(_resize_map(v, (tw, th), is_mask=True), 0.0, 1.0)
            for k, v in scene.instance_masks.items()
        },
        specular_color=_opt(scene.specular_color, clip01=True),
        material_confidence=_opt(scene.material_confidence, mask=True, clip01=True),
        material_metadata=dict(scene.material_metadata),
        albedo=_opt(scene.albedo, clip01=True),
        source_shading=(
            None
            if scene.source_shading is None
            else np.maximum(_resize_map(scene.source_shading, (tw, th)), 0.0)
        ),
        non_diffuse_residual=(
            None
            if scene.non_diffuse_residual is None
            else np.maximum(_resize_map(scene.non_diffuse_residual, (tw, th)), 0.0)
        ),
        intrinsic_metadata=dict(scene.intrinsic_metadata),
        raw_material_maps={
            k: (
                _resize_map(v, (tw, th))
                if k == "anisotropy_angle"
                else np.clip(_resize_map(v, (tw, th)), 0.0, 1.0)
            )
            for k, v in scene.raw_material_maps.items()
        },
        schema_version=scene.schema_version,
    )
    scaled_scene.validate()
    return scaled_scene, scaled_camera, scaled_arrays


def downsample_scene_buffers_for_preview(
    scene: SceneBuffers,
    camera: CameraModel,
    max_side: int,
) -> tuple[SceneBuffers, CameraModel]:
    """Proportionally downscale scene buffers and pinhole intrinsics for fast interactive preview."""
    scaled_scene, scaled_camera, _ = downsample_scene_and_arrays_for_preview(scene, camera, max_side)
    return scaled_scene, scaled_camera


def depth_guided_upsample_batch(
    low_fields: list[np.ndarray],
    depth_hi: np.ndarray,
    depth_lo: np.ndarray,
    *,
    depth_sigma: float = 0.04,
) -> list[np.ndarray]:
    """Batch 4-tap Joint Bilateral Upsample sharing log-depth affinity weights across multiple fields."""
    if not low_fields:
        return []
    h, w = depth_hi.shape
    lh, lw = depth_lo.shape
    if (lh, lw) == (h, w):
        return [f.astype(np.float32, copy=False) for f in low_fields]

    gx = (np.arange(w, dtype=np.float32) + 0.5) * (lw / float(w)) - 0.5
    gy = (np.arange(h, dtype=np.float32) + 0.5) * (lh / float(h)) - 0.5
    x0 = np.clip(np.floor(gx).astype(np.int32), 0, lw - 1)
    x1 = np.clip(x0 + 1, 0, lw - 1)
    wx = (gx - x0).astype(np.float32)[None, :]
    y0 = np.clip(np.floor(gy).astype(np.int32), 0, lh - 1)
    y1 = np.clip(y0 + 1, 0, lh - 1)
    wy = (gy - y0).astype(np.float32)[:, None]

    log_hi = np.log(np.maximum(depth_hi, 1e-3)).astype(np.float32)
    log_lo = np.log(np.maximum(depth_lo, 1e-3)).astype(np.float32)
    inv_2sig2 = np.float32(-0.5 / max(float(depth_sigma) ** 2, 1e-6))

    y0_ix = y0[:, None]
    y1_ix = y1[:, None]
    x0_ix = x0[None, :]
    x1_ix = x1[None, :]

    w00 = (1.0 - wy) * (1.0 - wx) * np.exp(np.maximum((log_hi - log_lo[y0_ix, x0_ix]) ** 2 * inv_2sig2, -20.0)) + np.float32(1e-6)
    w01 = (1.0 - wy) * wx * np.exp(np.maximum((log_hi - log_lo[y0_ix, x1_ix]) ** 2 * inv_2sig2, -20.0)) + np.float32(1e-6)
    w10 = wy * (1.0 - wx) * np.exp(np.maximum((log_hi - log_lo[y1_ix, x0_ix]) ** 2 * inv_2sig2, -20.0)) + np.float32(1e-6)
    w11 = wy * wx * np.exp(np.maximum((log_hi - log_lo[y1_ix, x1_ix]) ** 2 * inv_2sig2, -20.0)) + np.float32(1e-6)
    inv_den = np.float32(1.0) / (w00 + w01 + w10 + w11)

    outputs: list[np.ndarray] = []
    for f in low_fields:
        if f.ndim == 2:
            num = (
                w00 * f[y0_ix, x0_ix]
                + w01 * f[y0_ix, x1_ix]
                + w10 * f[y1_ix, x0_ix]
                + w11 * f[y1_ix, x1_ix]
            )
            outputs.append((num * inv_den).astype(np.float32))
        else:
            num = (
                w00[..., None] * f[y0_ix, x0_ix]
                + w01[..., None] * f[y0_ix, x1_ix]
                + w10[..., None] * f[y1_ix, x0_ix]
                + w11[..., None] * f[y1_ix, x1_ix]
            )
            outputs.append((num * inv_den[..., None]).astype(np.float32))
    return outputs


def depth_guided_upsample(
    low_field: np.ndarray,
    depth_hi: np.ndarray,
    depth_lo: np.ndarray,
    *,
    depth_sigma: float = 0.04,
) -> np.ndarray:
    """Fast 4-tap Joint Bilateral Upsample guided strictly by log-depth boundaries.

    Preserves sharp occlusion silhouettes across depth discontinuities while preventing
    texture/albedo bleed into lighting or visibility buffers.
    """
    return depth_guided_upsample_batch([low_field], depth_hi, depth_lo, depth_sigma=depth_sigma)[0]
