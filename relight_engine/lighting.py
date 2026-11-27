from __future__ import annotations
from collections import OrderedDict
import hashlib
import math
import os
import threading
import cv2
import numpy as np
from .color import luminance, srgb_to_linear
from .contracts import CameraModel, EnvironmentLight, Light, LightingRig, SceneBuffers, EngineSettings
from .shadows import scene_shadow_visibility

_ENV_IBL_CACHE: OrderedDict[tuple, tuple[list[np.ndarray], np.ndarray]] = OrderedDict()
_ENV_IBL_CACHE_LOCK = threading.Lock()
_ENV_IBL_CACHE_MAX = 4


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def _area_samples(light: Light, target_point: np.ndarray | None = None) -> list[np.ndarray]:
    center = np.asarray(light.position, np.float32)
    if light.direction is not None:
        toward_scene = np.asarray(light.direction, np.float32)
    else:
        target = np.array([0.0, 0.0, 1.0], np.float32) if target_point is None else np.asarray(target_point, np.float32)
        toward_scene = target - center
    norm = float(np.linalg.norm(toward_scene))
    if norm < 1e-6:
        toward_scene = np.array([0.0, 0.0, 1.0], np.float32)
    else:
        toward_scene /= norm
    up = np.array([0, 1, 0], np.float32)
    right = np.cross(up, toward_scene)
    if np.linalg.norm(right) < 1e-5:
        right = np.array([1, 0, 0], np.float32)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    up_axis = np.cross(toward_scene, right)
    up_axis /= max(float(np.linalg.norm(up_axis)), 1e-8)

    if light.samples <= 1:
        return [center]

    n = int(math.ceil(math.sqrt(light.samples)))
    # Centered, symmetric stratified bin centers in (-0.5, +0.5) so multi-sample grids are unbiased.
    offsets = (np.arange(n, dtype=np.float32) + 0.5) / float(n) - 0.5
    result = []
    for iy in range(n):
        for ix in range(n):
            if len(result) >= light.samples:
                break
            result.append(center + right * offsets[ix] * light.area_size[0] + up_axis * offsets[iy] * light.area_size[1])
    return result

def resolve_effective_materials(scene: SceneBuffers) -> dict[str, np.ndarray | None]:
    """Resolve effective rendering material properties with single-stage confidence blending.

    If `scene.material_metadata.get('confidence_applied')` is True, supplied maps were
    already blended in calibration and are not attenuated a second time by `material_confidence`.
    """
    h, w = scene.depth.shape
    priors = {
        "roughness": 0.75,
        "metalness": 0.0,
        "specular_level": 0.04,
        "specular_color": 0.65,
        "clearcoat": 0.0,
        "clearcoat_roughness": 0.15,
        "subsurface": 0.0,
    }
    already_applied = bool(scene.material_metadata.get("confidence_applied", False))
    confidence = (
        np.clip(np.asarray(scene.material_confidence, dtype=np.float32), 0.0, 1.0)
        if (scene.material_confidence is not None and not already_applied)
        else None
    )

    def resolve(name: str, supplied: np.ndarray | None, prior: float) -> np.ndarray:
        if supplied is None:
            return np.full((h, w), prior, dtype=np.float32) if name != "specular_color" else np.full((h, w, 3), prior, dtype=np.float32)
        values = np.asarray(supplied, dtype=np.float32)
        if name == "specular_color":
            if values.ndim == 2:
                values = np.repeat(values[..., None], 3, axis=-1)
            elif values.ndim == 1 and values.shape[0] == 3:
                values = np.broadcast_to(values, (h, w, 3)).copy()
        if confidence is None:
            return np.clip(values, 0.0, 1.0).astype(np.float32)
        weight = confidence if values.ndim == 2 else confidence[..., None]
        base = prior if values.ndim == 2 else np.full((1, 1, 3), prior, dtype=np.float32)
        return np.clip(base + weight * (values - base), 0.0, 1.0).astype(np.float32)

    aniso = None
    if scene.anisotropy is not None:
        aniso = np.clip(np.asarray(scene.anisotropy, dtype=np.float32), 0.0, 1.0)
        if confidence is not None:
            aniso = (aniso * confidence).astype(np.float32)

    aniso_angle = (
        np.asarray(scene.anisotropy_angle, dtype=np.float32)
        if scene.anisotropy_angle is not None
        else None
    )
    return {
        "roughness": resolve("roughness", scene.roughness, priors["roughness"]),
        "metalness": resolve("metalness", scene.metalness, priors["metalness"]),
        "specular_level": resolve("specular_level", scene.specular_level, priors["specular_level"]),
        "specular_color": resolve("specular_color", scene.specular_color, priors["specular_color"]),
        "clearcoat": resolve("clearcoat", scene.clearcoat, priors["clearcoat"]),
        "clearcoat_roughness": resolve("clearcoat_roughness", scene.clearcoat_roughness, priors["clearcoat_roughness"]),
        "subsurface": resolve("subsurface", scene.subsurface, priors["subsurface"]),
        "anisotropy": aniso,
        "anisotropy_angle": aniso_angle,
    }


def _effective_material_maps(
    scene: SceneBuffers,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return effective rendering maps resolved with single-stage confidence handling."""
    eff = resolve_effective_materials(scene)
    return (
        eff["roughness"],            # type: ignore[return-value]
        eff["metalness"],            # type: ignore[return-value]
        eff["specular_level"],       # type: ignore[return-value]
        eff["specular_color"],       # type: ignore[return-value]
        eff["clearcoat"],            # type: ignore[return-value]
        eff["clearcoat_roughness"],  # type: ignore[return-value]
        eff["subsurface"],           # type: ignore[return-value]
    )


def _effective_anisotropy_maps(scene: SceneBuffers) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve optional anisotropy magnitude [0, 1] (single-stage confidence-gated) and tangent angle."""
    eff = resolve_effective_materials(scene)
    return eff["anisotropy"], eff["anisotropy_angle"]


def _build_tangent_frame(
    normals: np.ndarray,
    angle: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal surface tangent/bitangent basis rotated optionally by anisotropy_angle."""
    up_ref = np.where(
        np.abs(normals[..., 1:2]) < 0.95,
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    t0 = _normalize(np.cross(up_ref, normals))
    b0 = _normalize(np.cross(normals, t0))
    if angle is None:
        return t0, b0
    c = np.cos(angle)[..., None]
    s = np.sin(angle)[..., None]
    t = _normalize(c * t0 + s * b0)
    b = _normalize(-s * t0 + c * b0)
    return t, b


def _ggx_specular(normals: np.ndarray, view: np.ndarray, light_dir: np.ndarray,
                  roughness: np.ndarray, metalness: np.ndarray,
                  specular_level: np.ndarray, specular_color: np.ndarray) -> np.ndarray:
    """GGX microfacet response with scalar dielectric F0 and RGB conductor F0."""
    ndotl = np.clip(np.sum(normals * light_dir, axis=-1), 0, 1)
    ndotv = np.clip(np.sum(normals * view, axis=-1), 1e-4, 1)
    half_vec = _normalize(light_dir + view)
    ndoth = np.clip(np.sum(normals * half_vec, axis=-1), 0, 1)
    vdoth = np.clip(np.sum(view * half_vec, axis=-1), 0, 1)
    r_eff = np.maximum(roughness, 0.045)
    alpha = r_eff ** 2
    alpha2 = alpha ** 2
    denom = np.maximum(np.pi * (ndoth * ndoth * (alpha2 - 1) + 1) ** 2, 1e-7)
    D = alpha2 / denom
    k = (r_eff + 1.0) ** 2 / 8.0
    Gv = ndotv / (ndotv * (1 - k) + k)
    Gl = ndotl / (ndotl * (1 - k) + k)
    G = Gv * Gl

    dielectric_f0 = np.clip(specular_level, 0.0, 1.0)[..., None]
    conductor_f0 = np.clip(specular_color, 0.0, 1.0)
    metal = np.clip(metalness, 0.0, 1.0)[..., None]
    f0 = dielectric_f0 * (1.0 - metal) + conductor_f0 * metal
    fresnel = f0 + (1.0 - f0) * (1.0 - vdoth[..., None]) ** 5
    scalar = (D * G * ndotl) / np.maximum(4.0 * ndotv * ndotl, 1e-5)
    return (scalar[..., None] * fresnel).astype(np.float32)

def _effective_specular_roughness(
    roughness: np.ndarray,
    light: Light,
    dist: np.ndarray | None,
) -> np.ndarray:
    """Broaden GGX roughness for finite light-source angular size on point/area lights."""
    if float(light.radius) <= 0.0 or dist is None:
        return roughness
    angular_r = np.clip(float(light.radius) / np.maximum(dist, 0.15), 0.0, 0.35)
    return np.sqrt(np.clip(roughness * roughness + 0.5 * angular_r, 0.045 * 0.045, 1.0)).astype(np.float32)


def _ggx_anisotropic_specular(
    normals: np.ndarray,
    view: np.ndarray,
    light_dir: np.ndarray,
    roughness: np.ndarray,
    metalness: np.ndarray,
    specular_level: np.ndarray,
    specular_color: np.ndarray,
    anisotropy: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
) -> np.ndarray:
    """Anisotropic GGX NDF reducing continuously to isotropic GGX at anisotropy=0."""
    ndotl = np.clip(np.sum(normals * light_dir, axis=-1), 0.0, 1.0)
    ndotv = np.clip(np.sum(normals * view, axis=-1), 1e-4, 1.0)
    half_vec = _normalize(light_dir + view)
    ndoth = np.clip(np.sum(normals * half_vec, axis=-1), 0.0, 1.0)
    vdoth = np.clip(np.sum(view * half_vec, axis=-1), 0.0, 1.0)
    tdoth = np.sum(tangent * half_vec, axis=-1)
    bdoth = np.sum(bitangent * half_vec, axis=-1)

    r_eff = np.maximum(roughness, 0.045)
    alpha = r_eff * r_eff
    aniso = np.clip(anisotropy, 0.0, 0.95)
    ax = np.maximum(alpha * (1.0 + aniso), 1e-4)
    ay = np.maximum(alpha * (1.0 - aniso), 1e-4)
    d_inner = (tdoth / ax) ** 2 + (bdoth / ay) ** 2 + ndoth ** 2
    D = 1.0 / np.maximum(np.pi * ax * ay * (d_inner ** 2), 1e-7)

    k = (r_eff + 1.0) ** 2 / 8.0
    Gv = ndotv / (ndotv * (1.0 - k) + k)
    Gl = ndotl / (ndotl * (1.0 - k) + k)
    G = Gv * Gl

    dielectric_f0 = np.clip(specular_level, 0.0, 1.0)[..., None]
    conductor_f0 = np.clip(specular_color, 0.0, 1.0)
    metal = np.clip(metalness, 0.0, 1.0)[..., None]
    f0 = dielectric_f0 * (1.0 - metal) + conductor_f0 * metal
    fresnel = f0 + (1.0 - f0) * (1.0 - vdoth[..., None]) ** 5
    scalar = (D * G * ndotl) / np.maximum(4.0 * ndotv * ndotl, 1e-5)
    return (scalar[..., None] * fresnel).astype(np.float32)


def _evaluate_specular_lobe(
    normals: np.ndarray,
    view: np.ndarray,
    light_dir: np.ndarray,
    roughness: np.ndarray,
    metalness: np.ndarray,
    specular_level: np.ndarray,
    specular_color: np.ndarray,
    *,
    clearcoat: np.ndarray | None = None,
    clearcoat_roughness: np.ndarray | None = None,
    has_coat: bool = False,
    anisotropy: np.ndarray | None = None,
    tangent: np.ndarray | None = None,
    bitangent: np.ndarray | None = None,
) -> np.ndarray:
    """Modular surface specular evaluator supporting isotropic/anisotropic GGX, metallic sheen, and clearcoat."""
    if anisotropy is not None and tangent is not None and bitangent is not None and np.any(anisotropy > 1e-4):
        spec = _ggx_anisotropic_specular(
            normals, view, light_dir, roughness, metalness, specular_level, specular_color,
            anisotropy, tangent, bitangent,
        )
    else:
        spec = _ggx_specular(
            normals, view, light_dir, roughness, metalness, specular_level, specular_color
        )
    if np.any(metalness > 1e-4):
        # Conductors exhibit a broader grazing sheen lobe in addition to the core GGX highlight.
        r_sheen = np.clip(roughness + 0.22, 0.08, 0.90).astype(np.float32)
        sheen = _ggx_specular(
            normals, view, light_dir, r_sheen, metalness, specular_level, specular_color
        )
        m_w = np.clip(metalness, 0.0, 1.0)[..., None]
        spec = spec * (1.0 - 0.25 * m_w) + (0.45 * m_w) * sheen
    if has_coat and clearcoat is not None and clearcoat_roughness is not None:
        h, w = roughness.shape
        zeros = np.zeros((h, w), dtype=np.float32)
        f04 = np.full((h, w), 0.04, dtype=np.float32)
        half_vec = _normalize(light_dir + view)
        vdoth = np.clip(np.sum(view * half_vec, axis=-1), 0.0, 1.0)
        coat_fresnel = 0.04 + 0.96 * np.power(1.0 - vdoth, 5.0)
        # Attenuate underlying base specular layer by clearcoat Fresnel reflection before adding top coat.
        coat_trans = 1.0 - clearcoat[..., None] * np.clip((coat_fresnel[..., None] - 0.04) * 2.5, 0.0, 0.75)
        coat_spec = _ggx_specular(
            normals, view, light_dir, clearcoat_roughness, zeros, f04, specular_color
        )
        spec = spec * coat_trans + clearcoat[..., None] * coat_spec
    return spec.astype(np.float32)


def _sample_equirectangular(
    env_img: np.ndarray,
    dirs: np.ndarray,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Bilinearly sample an equirectangular HxWx3 map with seamless horizontal wrap-around."""
    he, we = env_img.shape[:2]
    dx = dirs[..., 0]
    dy = np.clip(dirs[..., 1], -1.0, 1.0)
    dz = dirs[..., 2]
    phi = np.arctan2(dx, -dz)
    u = (phi / (2.0 * np.pi) + 0.5 + float(rotation_deg) / 360.0) % 1.0
    v = np.arccos(dy) / np.pi

    fx = u * we - 0.5
    fy = np.clip(v * (he - 1.0), 0.0, float(he - 1))
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    wx = (fx - x0)[..., None].astype(np.float32)
    wy = (fy - y0)[..., None].astype(np.float32)
    x0m = np.mod(x0, we)
    x1m = np.mod(x0 + 1, we)
    y0c = np.clip(y0, 0, he - 1)
    y1c = np.clip(y0 + 1, 0, he - 1)

    c00 = env_img[y0c, x0m]
    c10 = env_img[y0c, x1m]
    c01 = env_img[y1c, x0m]
    c11 = env_img[y1c, x1m]
    top = c00 * (1.0 - wx) + c10 * wx
    bot = c01 * (1.0 - wx) + c11 * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _prefilter_equirectangular_env(env_map: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Build (or fetch from content-keyed cache) roughness-prefiltered specular mip chain and diffuse irradiance."""
    buf = np.ascontiguousarray(np.maximum(np.asarray(env_map, dtype=np.float32), 0.0))
    key = (buf.shape, hashlib.blake2b(memoryview(buf), digest_size=16).digest())
    with _ENV_IBL_CACHE_LOCK:
        cached = _ENV_IBL_CACHE.get(key)
        if cached is not None:
            _ENV_IBL_CACHE.move_to_end(key)
            return cached

    env_lin = buf
    he, we = env_lin.shape[:2]
    # Preserve crisp mip 0 up to 256px wide; downsample wider maps so prefiltering stays sub-5ms.
    base_w = min(we, 256)
    base_h = max(16, int(round(he * (base_w / float(we)))))
    mip0 = (
        cv2.resize(env_lin, (base_w, base_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        if (we, he) != (base_w, base_h)
        else env_lin
    )
    blur_w, blur_h = 128, 64
    base_blur = cv2.resize(env_lin, (blur_w, blur_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    mips: list[np.ndarray] = [mip0]
    for r_level in (0.14, 0.30, 0.50, 0.72, 1.0):
        sig_x = max(0.7, (r_level ** 2) * (blur_w * 0.22))
        sig_y = max(0.5, (r_level ** 2) * (blur_h * 0.22))
        pad = min(blur_w // 2, int(math.ceil(2.5 * sig_x)))
        if pad > 0:
            padded = np.concatenate([base_blur[:, -pad:], base_blur, base_blur[:, :pad]], axis=1)
            blurred = cv2.GaussianBlur(padded, (0, 0), sigmaX=sig_x, sigmaY=sig_y, borderType=cv2.BORDER_REPLICATE)
            mips.append(blurred[:, pad:pad + blur_w].astype(np.float32))
        else:
            mips.append(cv2.GaussianBlur(base_blur, (0, 0), sigmaX=sig_x, sigmaY=sig_y, borderType=cv2.BORDER_REPLICATE).astype(np.float32))

    # Solid-angle weighted hemispherical cosine diffuse irradiance on a coarse spherical grid.
    ih, iw = 20, 40
    src_small = cv2.resize(env_lin, (iw, ih), interpolation=cv2.INTER_AREA).astype(np.float32)
    theta = (np.arange(ih, dtype=np.float32) + 0.5) * (np.pi / ih)
    phi = (np.arange(iw, dtype=np.float32) + 0.5 - 0.5 * iw) * (2.0 * np.pi / iw)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    sin_t = np.sin(tt)
    dx = sin_t * np.sin(pp)
    dy = np.cos(tt)
    dz = -sin_t * np.cos(pp)
    dirs = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
    domega = (sin_t * (np.pi / ih) * (2.0 * np.pi / iw)).reshape(-1)
    weighted_rad = (src_small.reshape(-1, 3) * domega[:, None]).astype(np.float32)
    cos_mat = np.clip(dirs @ dirs.T, 0.0, 1.0)
    irrad = ((cos_mat @ weighted_rad) / np.pi).reshape(ih, iw, 3).astype(np.float32)
    for m in mips:
        m.setflags(write=False)
    irrad.setflags(write=False)
    with _ENV_IBL_CACHE_LOCK:
        _ENV_IBL_CACHE[key] = (mips, irrad)
        _ENV_IBL_CACHE.move_to_end(key)
        while len(_ENV_IBL_CACHE) > _ENV_IBL_CACHE_MAX:
            _ENV_IBL_CACHE.popitem(last=False)
    return mips, irrad


def _sample_mip_chain(
    mips: list[np.ndarray],
    irrad_map: np.ndarray | None,
    dirs: np.ndarray,
    normals: np.ndarray,
    rough: np.ndarray,
    rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    n_mips = len(mips)
    # Nonlinear roughness-to-mip mapping preserves crisp mirror/water/clearcoat reflections.
    r_idx = np.power(np.clip(rough, 0.0, 1.0), 1.35) * float(n_mips - 1)
    i0 = np.floor(r_idx).astype(np.int32)
    i1 = np.minimum(i0 + 1, n_mips - 1)
    frac = (r_idx - i0)[..., None].astype(np.float32)

    sampled_levels = [
        _sample_equirectangular(mips[m], dirs, rotation_deg=rotation_deg)
        for m in range(n_mips)
    ]
    spec_env = np.zeros_like(normals, dtype=np.float32)
    for m in range(n_mips):
        w0 = ((i0 == m)[..., None]).astype(np.float32) * (1.0 - frac)
        w1 = ((i1 == m)[..., None]).astype(np.float32) * frac
        w_m = np.where((i0 == m)[..., None] & (i1 == m)[..., None], 1.0, w0 + w1)
        if np.any(w_m > 0.0):
            spec_env += sampled_levels[m] * w_m
    irrad_sampled = (
        _sample_equirectangular(irrad_map, normals, rotation_deg=rotation_deg)
        if irrad_map is not None
        else None
    )
    return spec_env, irrad_sampled


def evaluate_environment_lighting(
    normals: np.ndarray,
    view: np.ndarray,
    roughness: np.ndarray,
    metalness: np.ndarray,
    specular_level: np.ndarray,
    specular_color: np.ndarray,
    env: EnvironmentLight,
    *,
    anisotropy: np.ndarray | None = None,
    anisotropy_angle: np.ndarray | None = None,
    clearcoat: np.ndarray | None = None,
    clearcoat_roughness: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate analytic dome and/or equirectangular HDRI irradiance and Fresnel specular IBL."""
    env.validate()
    if anisotropy is not None and np.any(anisotropy > 1e-4):
        tangent, _ = _build_tangent_frame(normals, anisotropy_angle)
        aniso_w = (np.clip(anisotropy, 0.0, 0.95) * 0.45)[..., None]
        bent_n = _normalize(normals - tangent * (np.sum(tangent * view, axis=-1, keepdims=True) * aniso_w))
    else:
        bent_n = normals

    r_y = np.sum(bent_n * view, axis=-1, keepdims=True)
    refl = _normalize(2.0 * r_y * bent_n - view)
    ry = refl[..., 1]

    sky = np.asarray(env.sky_color, dtype=np.float32)[None, None, :]
    horizon = np.asarray(env.horizon_color, dtype=np.float32)[None, None, :]
    ground = np.asarray(env.ground_color, dtype=np.float32)[None, None, :]

    r_eff = np.clip(roughness, 0.045, 1.0)[..., None]
    env_radiance = np.zeros_like(normals, dtype=np.float32)
    irrad_total = np.zeros_like(normals, dtype=np.float32)
    coat_radiance = np.zeros_like(normals, dtype=np.float32)
    has_coat = clearcoat is not None and clearcoat_roughness is not None and bool(np.any(clearcoat > 1e-4))

    if float(env.strength) > 0.0:
        up_w = np.clip(ry, 0.0, 1.0)[..., None]
        dn_w = np.clip(-ry, 0.0, 1.0)[..., None]
        sharp_dome = (
            horizon * (1.0 - up_w - dn_w)
            + sky * up_w
            + ground * dn_w
        )
        dome_mean = 0.40 * sky + 0.40 * horizon + 0.20 * ground
        dome_rad = (sharp_dome * (1.0 - 0.75 * r_eff) + dome_mean * (0.75 * r_eff)) * float(env.strength)
        env_radiance = env_radiance + dome_rad
        if has_coat:
            assert clearcoat_roughness is not None
            rc_eff = np.clip(clearcoat_roughness, 0.045, 1.0)[..., None]
            coat_radiance = coat_radiance + (sharp_dome * (1.0 - 0.75 * rc_eff) + dome_mean * (0.75 * rc_eff)) * float(env.strength)
        sky_irrad_t = np.clip(0.5 * (normals[..., 1] + 1.0), 0.0, 1.0)[..., None]
        dome_irrad = ground * (1.0 - sky_irrad_t) + (0.6 * sky + 0.4 * horizon) * sky_irrad_t
        irrad_total = irrad_total + dome_irrad * (0.35 * float(env.strength))

    if float(env.key_strength) > 0.0:
        az = math.radians(float(env.key_azimuth_deg))
        el = math.radians(float(env.key_elevation_deg))
        key_dir = np.array(
            [math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)],
            dtype=np.float32,
        )
        key_dir /= max(float(np.linalg.norm(key_dir)), 1e-8)
        rdotl = np.clip(np.sum(refl * key_dir[None, None, :], axis=-1), 0.0, 1.0)
        eff_pow = np.maximum(1.5, float(env.key_sharpness) * (1.0 - 0.85 * roughness))
        key_lobe = np.power(rdotl, eff_pow) * (0.25 + 0.75 * (1.0 - roughness))
        key_col = np.asarray(env.key_color, dtype=np.float32)[None, None, :]
        env_radiance = env_radiance + key_col * (key_lobe[..., None] * float(env.key_strength))

    if env.env_map is not None and float(env.env_map_strength) > 0.0:
        mips, irrad_map = _prefilter_equirectangular_env(env.env_map)
        spec_env, irrad_sampled = _sample_mip_chain(
            mips, irrad_map, refl, normals, roughness, float(env.rotation_deg)
        )
        env_s = float(env.env_map_strength)
        env_radiance = env_radiance + spec_env * env_s
        if irrad_sampled is not None:
            irrad_total = irrad_total + irrad_sampled * env_s
        if has_coat:
            assert clearcoat_roughness is not None
            coat_spec, _ = _sample_mip_chain(
                mips, None, refl, normals, clearcoat_roughness, float(env.rotation_deg)
            )
            coat_radiance = coat_radiance + coat_spec * env_s

    ndotv = np.clip(np.sum(normals * view, axis=-1), 1e-4, 1.0)[..., None]
    dielectric_f0 = np.clip(specular_level, 0.0, 1.0)[..., None]
    conductor_f0 = np.clip(specular_color, 0.0, 1.0)
    metal = np.clip(metalness, 0.0, 1.0)[..., None]
    f0 = dielectric_f0 * (1.0 - metal) + conductor_f0 * metal
    # Roughness-attenuated Schlick Fresnel + quadratic roughness energy falloff keeps
    # matte surfaces free of washed sky glare while keeping smooth metals/water crisp.
    fresnel_rough = f0 + (np.maximum(1.0 - r_eff, f0) - f0) * np.power(1.0 - ndotv, 5.0)
    spec_rough_atten = np.clip(1.0 - 0.85 * (r_eff * r_eff), 0.10, 1.0)
    env_specular = env_radiance * fresnel_rough * spec_rough_atten

    if has_coat and clearcoat is not None and clearcoat_roughness is not None:
        rc_eff = np.clip(clearcoat_roughness, 0.045, 1.0)[..., None]
        coat_fresnel = 0.04 + (np.maximum(1.0 - rc_eff, 0.04) - 0.04) * np.power(1.0 - ndotv, 5.0)
        coat_spec = coat_radiance * coat_fresnel * np.clip(1.0 - 0.85 * (rc_eff * rc_eff), 0.10, 1.0)
        coat_trans = 1.0 - clearcoat[..., None] * np.clip(coat_fresnel * 2.5, 0.0, 0.75)
        env_specular = env_specular * coat_trans + clearcoat[..., None] * coat_spec

    env_specular = (env_specular * float(env.specular_strength)).astype(np.float32)
    coat_diff_atten = (1.0 - 0.15 * clearcoat[..., None] * (1.0 - ndotv)) if has_coat and clearcoat is not None else 1.0
    env_diffuse = (irrad_total * np.clip(1.0 - 0.88 * metal, 0.08, 1.0) * coat_diff_atten * float(env.diffuse_strength)).astype(np.float32)
    return env_diffuse, env_specular


def _light_contribution(scene: SceneBuffers, camera: CameraModel, points: np.ndarray,
                        normals: np.ndarray, light: Light, settings: EngineSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = scene.depth.shape
    color = np.asarray(light.color, np.float32)[None, None, :]
    (
        roughness,
        metalness,
        spec_level,
        specular_color,
        clearcoat,
        clearcoat_roughness,
        subsurface,
    ) = _effective_material_maps(scene)
    anisotropy, anisotropy_angle = _effective_anisotropy_maps(scene)
    has_aniso = anisotropy is not None and bool(np.any(anisotropy > 1e-4))
    tangent, bitangent = _build_tangent_frame(normals, anisotropy_angle) if has_aniso else (None, None)
    # Energy sharing between diffuse and Fresnel specular + grazing clearcoat transmission.
    diffuse_energy = np.clip(1.0 - 0.88 * metalness, 0.08, 1.0).astype(np.float32)[..., None]
    diffuse_energy = diffuse_energy * (1.0 - 0.5 * np.clip(spec_level, 0.0, 0.35)[..., None])
    has_coat = bool(np.any(clearcoat > 1e-4))
    has_sss = bool(np.any(subsurface > 1e-4)) and os.environ.get("RELIGHT_ABLATE_NO_SSS_WRAP") != "1"
    view = _normalize(-points)
    if has_coat:
        ndotv_coat = np.clip(np.sum(normals * view, axis=-1), 0.0, 1.0)
        diffuse_energy = diffuse_energy * (1.0 - 0.25 * clearcoat * (1.0 - ndotv_coat))[..., None]
    diffuse = np.zeros((h, w, 3), np.float32)
    specular = np.zeros((h, w, 3), np.float32)
    visibility_sum = np.zeros((h, w), np.float32)

    eff_strength = (
        float(light.strength) / math.pi
        if (light.intensity_mode == 'physical' and light.kind in ('point', 'area', 'spot'))
        else float(light.strength)
    )
    eff_radius = float(light.radius)
    if light.kind == 'area':
        eff_radius = max(eff_radius, 0.35 * math.hypot(float(light.area_size[0]), float(light.area_size[1])))

    spot_dir = None
    spot_cos_in = 1.0
    spot_cos_out = -1.0
    if light.kind == 'directional':
        toward = np.asarray(light.direction, np.float32)
        toward /= max(float(np.linalg.norm(toward)), 1e-8)
        dirs = [np.broadcast_to(toward, (h, w, 3))]
        positions = [None]
    elif light.kind in ('point', 'spot'):
        lp = np.asarray(light.position, np.float32)
        vec = lp[None, None, :] - points
        dist = np.linalg.norm(vec, axis=-1)
        dirs = [vec / np.maximum(dist[..., None], 1e-8)]
        positions = [np.broadcast_to(lp, (h, w, 3))]
        if light.kind == 'spot':
            assert light.direction is not None
            sd = np.asarray(light.direction, np.float32)
            spot_dir = sd / max(float(np.linalg.norm(sd)), 1e-8)
            spot_cos_in = math.cos(math.radians(float(light.spot_inner_angle_deg)))
            spot_cos_out = math.cos(math.radians(float(light.spot_outer_angle_deg)))
    else:
        valid = scene.depth > 0
        scene_target = np.mean(points[valid], axis=0).astype(np.float32) if np.any(valid) else None
        positions = [np.broadcast_to(p, (h, w, 3)) for p in _area_samples(light, target_point=scene_target)]
        dirs = []
        for pos in positions:
            vec = pos - points
            dist = np.linalg.norm(vec, axis=-1)
            dirs.append(vec / np.maximum(dist[..., None], 1e-8))

    count = len(dirs)
    for d, pos in zip(dirs, positions):
        if light.kind == 'directional':
            dist = None
        else:
            dist = np.linalg.norm(pos - points, axis=-1)
        raw_ndotl = np.sum(normals * d, axis=-1)
        ndotl = np.clip(raw_ndotl, 0, 1)
        if light.wrap > 0.0:
            w_fac = float(light.wrap)
            ndotl_diffuse = np.clip((raw_ndotl + w_fac) / (1.0 + w_fac), 0, 1)
        else:
            ndotl_diffuse = ndotl
        if light.kind == 'directional':
            attenuation = np.full((h, w), eff_strength, np.float32)
        elif scene.depth_kind == 'metric':
            attenuation = eff_strength / np.maximum(dist * dist, 0.04)
        else:
            # Normalize distance by both median scene depth and light standoff distance so
            # close studio lights keep natural local falloff while distant sky/sun-scale
            # landscape lights illuminate valley and peaks evenly without starving the horizon.
            valid_d = scene.depth > 0
            z_ref = max(float(np.median(scene.depth[valid_d])) if np.any(valid_d) else 2.0, 1.0)
            dist_ref = float(np.median(dist[valid_d])) if np.any(valid_d) else z_ref
            rel_scale = max(0.45 * z_ref, 0.48 * dist_ref, 1.0)
            d_scaled = dist / rel_scale
            attenuation = eff_strength / (1.0 + 0.30 * d_scaled * d_scaled)
        if spot_dir is not None:
            cos_theta = np.sum((-d) * spot_dir[None, None, :], axis=-1)
            t_cone = np.clip((cos_theta - spot_cos_out) / max(spot_cos_in - spot_cos_out, 1e-5), 0.0, 1.0)
            attenuation = attenuation * (t_cone * t_cone * (3.0 - 2.0 * t_cone)).astype(np.float32)
        if light.casts_shadows:
            if getattr(settings, "shadow_half_res", False) and max(h, w) > 512:
                scale_s = 448.0 / float(max(h, w))
                hs = max(32, int(round(h * scale_s)))
                ws = max(32, int(round(w * scale_s)))
                from .preview import depth_guided_upsample_batch
                sc_s = SceneBuffers(
                    rgb_srgb=cv2.resize(scene.rgb_srgb, (ws, hs), interpolation=cv2.INTER_AREA),
                    depth=cv2.resize(scene.depth, (ws, hs), interpolation=cv2.INTER_NEAREST),
                    depth_confidence=cv2.resize(scene.depth_confidence, (ws, hs), interpolation=cv2.INTER_AREA),
                    lightable_mask=cv2.resize(scene.lightable_mask, (ws, hs), interpolation=cv2.INTER_AREA),
                    depth_kind=scene.depth_kind,
                    semantic_masks={
                        k: cv2.resize(m, (ws, hs), interpolation=cv2.INTER_AREA)
                        for k, m in scene.semantic_masks.items()
                        if any(t in k.lower() for t in ("sky", "cloud", "heaven"))
                    },
                )
                cam_s = CameraModel(
                    camera.fx * (ws / float(w)),
                    camera.fy * (hs / float(h)),
                    camera.cx * (ws / float(w)),
                    camera.cy * (hs / float(h)),
                )
                pts_s = cv2.resize(points, (ws, hs), interpolation=cv2.INTER_NEAREST)
                nrm_s = _normalize(cv2.resize(normals, (ws, hs), interpolation=cv2.INTER_NEAREST))
                pos_s = np.broadcast_to(pos[:1, :1, :], (hs, ws, 3)) if pos is not None else None
                dir_s = (
                    np.broadcast_to(d[:1, :1, :], (hs, ws, 3))
                    if (pos is None and light.kind == 'directional')
                    else (cv2.resize(d, (ws, hs), interpolation=cv2.INTER_AREA) if pos is None else None)
                )
                vis_lo = scene_shadow_visibility(
                    sc_s, cam_s, pts_s, nrm_s, pos_s, dir_s, settings, light_radius=eff_radius,
                )
                visibility = np.clip(
                    depth_guided_upsample_batch([vis_lo], scene.depth, sc_s.depth, depth_sigma=0.025)[0], 0.0, 1.0
                ).astype(np.float32)
            else:
                visibility = scene_shadow_visibility(
                    scene, camera, points, normals, pos,
                    d if light.kind == 'directional' else None,
                    settings, light_radius=eff_radius,
                )
        else:
            visibility = np.ones((h, w), np.float32)
        if has_sss:
            # Energy-conserving per-channel RGB wrapped diffuse for realistic skin terminator scatter.
            w_r = float(light.wrap) + 0.24 * subsurface
            w_g = float(light.wrap) + 0.09 * subsurface
            w_b = float(light.wrap) + 0.02 * subsurface
            ndotl_r = np.clip((raw_ndotl + w_r) / (1.0 + w_r), 0.0, 1.0)
            ndotl_g = np.clip((raw_ndotl + w_g) / (1.0 + w_g), 0.0, 1.0)
            ndotl_b = np.clip((raw_ndotl + w_b) / (1.0 + w_b), 0.0, 1.0)
            ndotl_rgb = np.stack([ndotl_r, ndotl_g, ndotl_b], axis=-1).astype(np.float32)
            diffuse += ndotl_rgb * ((attenuation * visibility / count)[..., None] * color) * diffuse_energy
        else:
            direct = (ndotl_diffuse * attenuation * visibility / count).astype(np.float32)
            diffuse += (direct[..., None] * color) * diffuse_energy
        r_spec = _effective_specular_roughness(roughness, light, dist)
        spec = _evaluate_specular_lobe(
            normals, view, d, r_spec, metalness, spec_level, specular_color,
            clearcoat=clearcoat, clearcoat_roughness=clearcoat_roughness, has_coat=has_coat,
            anisotropy=anisotropy, tangent=tangent, bitangent=bitangent,
        )
        specular += spec * (attenuation * visibility / count)[..., None] * color
        visibility_sum += visibility / count
    return diffuse, specular, visibility_sum


def compute_indirect_bounce(
    scene: SceneBuffers,
    points: np.ndarray,
    normals: np.ndarray,
    direct_diffuse: np.ndarray,
    settings: EngineSettings,
) -> np.ndarray:
    """Multi-scale 3D global VPL surfel radiosity + local ring inter-surface bounce (SSGI).

    Propagates colored direct illumination across the entire 3D scene (floor <-> subject <-> wall)
    using two-sided emitter/receiver normal form factors and depth-scaled emitter areas, while
    excluding infinite sky pixels when `settings.protect_sky` is enabled.
    """
    strength = float(settings.indirect_bounce_strength)
    if strength <= 0.0:
        return np.zeros_like(direct_diffuse, dtype=np.float32)

    h, w = scene.depth.shape
    if scene.albedo is not None:
        albedo_lin = srgb_to_linear(scene.albedo)
    else:
        # Without intrinsic albedo, desaturate source RGB toward neutral luminance so source casts don't bleed.
        src_lin = srgb_to_linear(scene.rgb_srgb)
        src_y = luminance(src_lin)[..., None]
        albedo_lin = np.clip(0.65 * src_y + 0.35 * src_lin, 0.04, 0.65).astype(np.float32)

    # Softly bound bounce radiance so increasing light count cannot cause indirect energy blowup.
    bounded_direct = np.maximum(direct_diffuse, 0.0)
    bounded_direct = bounded_direct / (1.0 + 0.25 * bounded_direct)
    radiance = (albedo_lin * bounded_direct).astype(np.float32)

    valid = scene.depth > 0
    if getattr(settings, "protect_sky", True):
        from .scene_policy import detect_sky_horizon_mask
        sky_w = detect_sky_horizon_mask(scene)
        non_sky = (1.0 - sky_w).astype(np.float32)
        radiance = radiance * non_sky[..., None]
        valid_emit = valid & (sky_w < 0.25)
    else:
        non_sky = np.ones((h, w), dtype=np.float32)
        valid_emit = valid

    log_d = np.log(np.maximum(scene.depth, 1e-4)).astype(np.float32)
    z_med = max(float(np.median(scene.depth[valid])) if np.any(valid) else 1.0, 1e-3)
    world_scale = max(0.55 * z_med, 0.65)

    # 1. Global full-scene VPL surfel radiosity gather across downsampled 3D surfels (fast CPU path).
    global_bounce = np.zeros_like(radiance, dtype=np.float32)
    if np.any(valid_emit):
        # Emitter grid: ~16x16 surfels (<= 256 emitters covering full room/floor/subject).
        eh = max(4, min(16, h))
        ew = max(4, min(16, w))
        emit_rad = cv2.resize(radiance, (ew, eh), interpolation=cv2.INTER_AREA).reshape(-1, 3)
        emit_pts = cv2.resize(points, (ew, eh), interpolation=cv2.INTER_AREA).reshape(-1, 3)
        emit_nrm = _normalize(cv2.resize(normals, (ew, eh), interpolation=cv2.INTER_AREA)).reshape(-1, 3)
        emit_val = (cv2.resize(valid_emit.astype(np.float32), (ew, eh), interpolation=cv2.INTER_AREA).ravel() > 0.25)
        if np.any(emit_val):
            e_rad = emit_rad[emit_val]
            e_pts = emit_pts[emit_val]
            e_nrm = emit_nrm[emit_val]
            e_area = np.clip((np.maximum(e_pts[:, 2], 0.1) / z_med) ** 2, 0.25, 4.0).astype(np.float32)

            # Receiver grid capped at <= 128 px max side, then depth-aware upsampled cleanly.
            scale_r = min(1.0, 128.0 / float(max(h, w)))
            rh = max(4, int(round(h * scale_r)))
            rw = max(4, int(round(w * scale_r)))
            r_pts = cv2.resize(points, (rw, rh), interpolation=cv2.INTER_AREA).reshape(-1, 3)
            r_nrm = _normalize(cv2.resize(normals, (rw, rh), interpolation=cv2.INTER_AREA)).reshape(-1, 3)
            r_val = (cv2.resize(valid.astype(np.float32), (rw, rh), interpolation=cv2.INTER_AREA).ravel() > 0.25)

            delta = e_pts[None, :, :] - r_pts[:, None, :]  # (R, E, 3)
            dist2 = np.sum(delta * delta, axis=-1)
            dist = np.sqrt(np.maximum(dist2, 1e-6))
            d_hat = delta / dist[..., None]

            cos_r = np.clip(np.sum(r_nrm[:, None, :] * d_hat, axis=-1), 0.08, 1.0)
            cos_e = np.clip(np.sum(e_nrm[None, :, :] * (-d_hat), axis=-1), 0.15, 1.0)
            soft2 = (0.18 * world_scale) ** 2
            att = 1.0 / (1.0 + (dist2 + soft2) / max(world_scale * world_scale, 1e-4))
            # Allow cross-object bounce between floor/subject/wall while gently tapering massive horizon chasms.
            log_gap = np.abs(
                np.log(np.maximum(e_pts[None, :, 2], 0.1)) - np.log(np.maximum(r_pts[:, None, 2], 0.1))
            )
            gap_atten = np.where(log_gap <= 1.25, 1.0, np.exp(-(log_gap - 1.25) * 1.5))

            w_mat = (cos_r * cos_e * e_area[None, :] * att * gap_atten * r_val[:, None]).astype(np.float32)
            w_tot = np.maximum(np.sum(w_mat, axis=1, keepdims=True), 1e-4)
            g_lo = ((w_mat @ e_rad) / w_tot).reshape(rh, rw, 3).astype(np.float32)
            global_bounce = cv2.resize(g_lo, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # 2. Local multi-radius ring for crisp contact corners (under-chin, floor-wall seam, collar).
    r_max = max(2, int(round(settings.indirect_bounce_radius)))
    r_mid = max(1, r_max // 2)
    r_near = max(1, r_max // 4)
    offsets = []
    for rad in (r_near, r_mid, r_max):
        diag = max(1, int(round(rad * 0.7071)))
        offsets.extend([
            (0, rad), (0, -rad), (rad, 0), (-rad, 0),
            (diag, diag), (diag, -diag), (-diag, diag), (-diag, -diag),
        ])

    accum = np.zeros_like(radiance, dtype=np.float32)
    w_sum = np.zeros((h, w), dtype=np.float32)
    for dy, dx in offsets:
        rad_s = np.roll(np.roll(radiance, dy, axis=0), dx, axis=1)
        pts_s = np.roll(np.roll(points, dy, axis=0), dx, axis=1)
        nrm_s = np.roll(np.roll(normals, dy, axis=0), dx, axis=1)
        log_s = np.roll(np.roll(log_d, dy, axis=0), dx, axis=1)
        val_s = np.roll(np.roll(valid_emit, dy, axis=0), dx, axis=1).copy()
        if dy > 0:
            val_s[:dy, :] = False
        elif dy < 0:
            val_s[dy:, :] = False
        if dx > 0:
            val_s[:, :dx] = False
        elif dx < 0:
            val_s[:, dx:] = False

        delta = pts_s - points
        dist2 = np.sum(delta * delta, axis=-1)
        dist = np.sqrt(np.maximum(dist2, 1e-8))
        dir_to_src = delta / dist[..., None]
        facing = np.clip(np.sum(normals * dir_to_src, axis=-1), 0.10, 1.0)
        emit_facing = np.clip(np.sum(nrm_s * (-dir_to_src), axis=-1), 0.15, 1.0)
        # Allow cross-object corner bounce across typical studio depth steps (~0.85 log units).
        depth_weight = np.exp(-np.abs(log_s - log_d) / 0.85)
        dist_weight = 1.0 / (1.0 + dist2 / max(world_scale * world_scale, 1e-4))
        weight = np.where(valid & val_s, facing * emit_facing * depth_weight * dist_weight, 0.0).astype(np.float32)
        accum += rad_s * weight[..., None]
        w_sum += weight

    local_bounce = accum / np.maximum(w_sum[..., None], 1.0)
    combined = (0.60 * global_bounce + 0.40 * local_bounce) * non_sky[..., None]
    # Soft energy conservation cap keeps shadow lift natural without washing contrast.
    combined = combined / (1.0 + 0.35 * combined)
    return (combined * strength).astype(np.float32)


def compute_volumetric_beam(
    scene: SceneBuffers,
    points: np.ndarray,
    rig: LightingRig,
    visibility: np.ndarray,
    settings: EngineSettings,
) -> np.ndarray:
    """Subtle depth- and occlusion-aware 3D radial air glow around active point and spot studio lights."""
    strength = float(getattr(settings, "volumetric_beam_strength", 0.0))
    if strength <= 0.0 or not rig.lights:
        return np.zeros((*scene.shape, 3), dtype=np.float32)

    from .scene_policy import classify_scene_scale
    scale_info = classify_scene_scale(scene)
    if bool(scale_info["is_landscape"]):
        # Suppress artificial foreground studio fog pools/cones on macroscopic mountain/landscape scenes.
        return np.zeros((*scene.shape, 3), dtype=np.float32)

    h, w = scene.shape
    valid = scene.depth > 0
    z_med = max(float(np.median(scene.depth[valid])) if np.any(valid) else 2.0, 1.0)
    scale = max(0.65 * z_med, 0.8)
    vis_gate = (0.30 + 0.70 * np.clip(visibility, 0.0, 1.0))[..., None]

    glow = np.zeros((h, w, 3), dtype=np.float32)
    for light in rig.lights:
        if light.kind not in ("point", "spot", "area") or light.position is None:
            continue
        lp = np.asarray(light.position, dtype=np.float32)
        vec = points - lp[None, None, :]
        dist2 = np.sum(vec * vec, axis=-1)
        radial = 1.0 / (1.0 + dist2 / (scale * scale))
        if light.kind == "spot" and light.direction is not None:
            sd = np.asarray(light.direction, dtype=np.float32)
            sd = sd / max(float(np.linalg.norm(sd)), 1e-8)
            d_ray = vec / np.maximum(np.sqrt(dist2)[..., None], 1e-8)
            cos_t = np.sum(d_ray * sd[None, None, :], axis=-1)
            cos_in = math.cos(math.radians(float(light.spot_inner_angle_deg)))
            cos_out = math.cos(math.radians(float(light.spot_outer_angle_deg)))
            cone = np.clip((cos_t - cos_out) / max(cos_in - cos_out, 1e-5), 0.0, 1.0)
            radial = radial * (cone * cone)
        col = np.asarray(light.color, dtype=np.float32)[None, None, :]
        glow += (radial[..., None] * col) * (0.20 * min(float(light.strength), 4.0) * strength)

    glow = glow * vis_gate
    if getattr(settings, "protect_sky", True):
        from .scene_policy import detect_sky_horizon_mask
        sky_w = detect_sky_horizon_mask(scene)
        glow = glow * (1.0 - sky_w)[..., None]
    return np.clip(glow, 0.0, 0.45).astype(np.float32)


def compute_indirect_specular(
    scene: SceneBuffers,
    points: np.ndarray,
    normals: np.ndarray,
    bounce_diffuse: np.ndarray,
    settings: EngineSettings,
) -> np.ndarray:
    """Apply Fresnel-weighted bounced room radiance to glossy/metallic/clearcoat surfaces."""
    strength = float(getattr(settings, "indirect_specular_strength", 0.0))
    if strength <= 0.0 or not np.any(bounce_diffuse > 1e-6):
        return np.zeros_like(bounce_diffuse, dtype=np.float32)

    roughness, metalness, spec_level, specular_color, clearcoat, clearcoat_roughness, _ = _effective_material_maps(scene)
    view = _normalize(-points)
    ndotv = np.clip(np.sum(normals * view, axis=-1), 1e-4, 1.0)[..., None]

    f0_d = np.clip(spec_level, 0.0, 1.0)[..., None]
    f0_m = np.clip(specular_color, 0.0, 1.0)
    m_w = np.clip(metalness, 0.0, 1.0)[..., None]
    f0 = f0_d * (1.0 - m_w) + f0_m * m_w
    fresnel = f0 + (np.maximum(1.0 - roughness[..., None], f0) - f0) * np.power(1.0 - ndotv, 5.0)
    gloss_w = np.clip(1.0 - 0.78 * (roughness * roughness), 0.08, 1.0)[..., None]
    spec_bounce = bounce_diffuse * fresnel * gloss_w

    if bool(np.any(clearcoat > 1e-4)):
        rc = np.clip(clearcoat_roughness, 0.045, 1.0)[..., None]
        coat_f = 0.04 + (np.maximum(1.0 - rc, 0.04) - 0.04) * np.power(1.0 - ndotv, 5.0)
        spec_bounce = spec_bounce + bounce_diffuse * coat_f * clearcoat[..., None] * 0.65

    return (spec_bounce * strength).astype(np.float32)


def _evaluate_virtual_studio_reflection(
    scene: SceneBuffers,
    points: np.ndarray,
    normals: np.ndarray,
    rig: LightingRig,
    settings: EngineSettings,
    roughness: np.ndarray,
    metalness: np.ndarray,
    spec_level: np.ndarray,
    specular_color: np.ndarray,
    clearcoat: np.ndarray,
    clearcoat_roughness: np.ndarray,
) -> np.ndarray:
    """Synthesize view-dependent reflection transport along R = reflect(-V, N) when no HDRI env is bound."""
    h, w = scene.depth.shape
    if not rig.lights:
        return np.zeros((h, w, 3), dtype=np.float32)

    view = _normalize(-points)
    ndotv = np.clip(np.sum(normals * view, axis=-1), 0.0, 1.0)
    # R = 2*(N.V)*N - V
    refl = _normalize(2.0 * ndotv[..., None] * normals - view)

    f0_dielectric = (0.04 + 0.12 * np.clip(spec_level, 0.0, 1.0))[..., None]
    f0 = specular_color * metalness[..., None] + f0_dielectric * (1.0 - metalness[..., None])
    fresnel = f0 + (1.0 - f0) * np.power(1.0 - ndotv[..., None], 5.0)
    has_coat = bool(np.any(clearcoat > 1e-4))
    coat_fresnel = (0.04 + 0.96 * np.power(1.0 - ndotv, 5.0)) * clearcoat if has_coat else 0.0

    valid_d = scene.depth > 0
    z_ref = max(float(np.median(scene.depth[valid_d])) if np.any(valid_d) else 2.0, 1.0)
    self_occ = np.clip(np.sum(refl * normals, axis=-1) + 0.15, 0.0, 1.0)
    horizon_gate = (0.35 + 0.65 * self_occ)[..., None].astype(np.float32)
    rough_energy = np.clip(1.0 - 0.55 * roughness, 0.15, 1.0)[..., None].astype(np.float32)

    total_refl = np.zeros((h, w, 3), dtype=np.float32)
    for light in rig.lights:
        col = np.asarray(light.color, dtype=np.float32)[None, None, :]
        eff_str = (
            float(light.strength) / math.pi
            if (light.intensity_mode == 'physical' and light.kind in ('point', 'area', 'spot'))
            else float(light.strength)
        )
        eff_radius = max(float(light.radius), 0.25)
        if light.kind == 'area':
            eff_radius = max(eff_radius, 0.35 * math.hypot(float(light.area_size[0]), float(light.area_size[1])))

        if light.kind == 'directional':
            toward = np.asarray(light.direction, dtype=np.float32)
            toward /= max(float(np.linalg.norm(toward)), 1e-8)
            ldir = np.broadcast_to(toward, (h, w, 3))
            atten = np.full((h, w), eff_str, dtype=np.float32)
            ang_light = 0.18
        else:
            lp = np.asarray(light.position, dtype=np.float32)
            vec = lp[None, None, :] - points
            dist = np.linalg.norm(vec, axis=-1)
            ldir = vec / np.maximum(dist[..., None], 1e-8)
            if scene.depth_kind == 'metric':
                atten = eff_str / np.maximum(dist * dist, 0.04)
            else:
                dist_ref = float(np.median(dist[valid_d])) if np.any(valid_d) else z_ref
                rel_scale = max(0.45 * z_ref, 0.48 * dist_ref, 1.0)
                d_scaled = dist / rel_scale
                atten = eff_str / (1.0 + 0.30 * d_scaled * d_scaled)
            ang_light = np.arctan(eff_radius / np.maximum(dist, 0.25))

        cos_r = np.clip(np.sum(refl * ldir, axis=-1), -1.0, 1.0)
        angle = np.arccos(cos_r)
        alpha = np.maximum(ang_light + 0.12 + 0.72 * roughness, 0.10)
        lobe = np.exp(-0.5 * (angle / alpha) ** 2).astype(np.float32)
        radiance = col * (atten * lobe)[..., None]

        contrib = radiance * fresnel * rough_energy
        if has_coat:
            alpha_c = np.maximum(ang_light + 0.08 + 0.65 * clearcoat_roughness, 0.08)
            lobe_c = np.exp(-0.5 * (angle / alpha_c) ** 2).astype(np.float32)
            contrib = contrib + col * (atten * lobe_c * coat_fresnel)[..., None]
        total_refl += contrib

    return (total_refl * horizon_gate * float(settings.specular_scale)).astype(np.float32)


def evaluate_rig(scene: SceneBuffers, camera: CameraModel, points: np.ndarray,
                 normals: np.ndarray, rig: LightingRig, settings: EngineSettings,
                 *, return_details: bool = False):
    rig.validate()
    h, w = scene.depth.shape
    roughness, metalness, spec_level, specular_color, clearcoat, clearcoat_roughness, _ = _effective_material_maps(scene)
    # Without an HDRI environment map, discrete studio lights miss broad grazing solid angles
    # that keep polished/brushed metals readable; keep a bounded diffuse floor when no env map is bound.
    metal_cut = 0.88 if rig.environment is not None else 0.58
    diffuse_energy = np.clip(1.0 - metal_cut * metalness, 0.08 if rig.environment is not None else 0.32, 1.0).astype(np.float32)[..., None]
    diffuse_energy = diffuse_energy * (1.0 - 0.5 * np.clip(spec_level, 0.0, 0.35)[..., None])
    if bool(np.any(clearcoat > 1e-4)):
        view_amb = _normalize(-points)
        ndotv_amb = np.clip(np.sum(normals * view_amb, axis=-1), 0.0, 1.0)
        diffuse_energy = diffuse_energy * (1.0 - 0.25 * clearcoat * (1.0 - ndotv_amb))[..., None]
    fill_bal = float(np.clip(getattr(settings, "preserve_fill_balance", 0.0), 0.0, 1.0))
    eff_amb_str = rig.ambient_strength
    if fill_bal > 0.0 and rig.lights:
        # Preserve soft omnidirectional room fill on grazing/crevice surfaces when replacing source light.
        eff_amb_str = max(rig.ambient_strength, rig.ambient_strength + 0.24 * fill_bal)

    sky_color = np.asarray(rig.ambient_color, np.float32)[None, None, :]
    if rig.ambient_ground_color is not None:
        ground_color = np.asarray(rig.ambient_ground_color, np.float32)[None, None, :]
        sky_t = np.clip(0.5 * (normals[..., 1] + 1.0), 0.0, 1.0)[..., None]
        ambient = (ground_color * (1.0 - sky_t) + sky_color * sky_t) * eff_amb_str
    elif (float(settings.indirect_bounce_strength) > 0.0 or fill_bal > 0.0) and rig.lights:
        # Subtle hemisphere studio fill weighted toward light warmth for downward-facing undersides.
        l_cols = np.mean([np.asarray(lt.color, np.float32) for lt in rig.lights], axis=0)[None, None, :]
        l_cols = l_cols / max(float(np.max(l_cols)), 1e-4)
        ground_color = (0.72 * sky_color + 0.28 * l_cols).astype(np.float32)
        sky_t = np.clip(0.5 * (normals[..., 1] + 1.0), 0.0, 1.0)[..., None]
        ambient = (ground_color * (1.0 - sky_t) + sky_color * sky_t) * eff_amb_str
    else:
        ambient = sky_color * eff_amb_str

    eff_ao = 1.0
    ssao_s = float(getattr(settings, "ssao_strength", 0.0))
    if ssao_s > 0.0:
        from .geometry import compute_screen_space_ao
        ao_map = compute_screen_space_ao(points, normals, scene.depth)
        if scene.source_shading is not None:
            # Suppress redundant SSAO darkening inside already-dark baked source crevices (e.g. finger gaps).
            sh_arr = scene.source_shading if scene.source_shading.ndim == 2 else luminance(scene.source_shading)
            ao_gate = np.clip((sh_arr - 0.14) / 0.22, 0.0, 1.0).astype(np.float32)
        else:
            ao_gate = 1.0
        eff_ao = (1.0 - ssao_s * ao_gate * (1.0 - ao_map))[..., None].astype(np.float32)

    ambient_term = (np.broadcast_to(ambient, (h, w, 3)).astype(np.float32, copy=True) * diffuse_energy * eff_ao).astype(np.float32)

    direct_diffuse = np.zeros((h, w, 3), np.float32)
    specular = np.zeros((h, w, 3), np.float32)
    ind_spec_s = float(getattr(settings, "indirect_specular_strength", 0.0))
    if rig.environment is None and ind_spec_s > 0.0 and rig.lights and np.any(metalness > 0.25):
        # Provide gentle room-ambient metallic reflection when no explicit HDRI environment map is bound
        # so bare steel/jewelry does not drop abnormally dark between discrete point-light specular peaks.
        amb_gloss = np.clip(1.0 - 0.55 * roughness, 0.15, 1.0).astype(np.float32)
        amb_metal_w = np.clip((metalness - 0.15) / 0.85, 0.0, 1.0).astype(np.float32)
        specular += (
            np.broadcast_to(ambient, (h, w, 3))
            * specular_color
            * (amb_metal_w * amb_gloss * min(ind_spec_s, 0.65))[..., None]
            * eff_ao
        ).astype(np.float32)
    if rig.environment is None and getattr(settings, "virtual_studio_reflection", False) and rig.lights:
        specular += _evaluate_virtual_studio_reflection(
            scene,
            points,
            normals,
            rig,
            settings,
            roughness,
            metalness,
            spec_level,
            specular_color,
            clearcoat,
            clearcoat_roughness,
        ) * eff_ao
    visibility = np.ones((h, w), np.float32)
    vis_weight_sum = np.zeros((h, w), np.float32)
    vis_weighted_acc = np.zeros((h, w), np.float32)
    per_light_records: list[dict] = []
    per_light_vis: dict[str, np.ndarray] = {}

    for light in rig.lights:
        d, s, v = _light_contribution(scene, camera, points, normals, light, settings)
        direct_diffuse += d
        specular += s
        visibility = np.minimum(visibility, v)
        d_mag = luminance(d) + 1e-4
        vis_weighted_acc += v * d_mag
        vis_weight_sum += d_mag
        lid = light.resolved_id
        per_light_vis[lid] = v
        per_light_records.append({
            "light_id": lid,
            "name": light.name,
            "kind": light.kind,
            "intensity_mode": light.intensity_mode,
            "diffuse_mean": float(np.mean(d)),
            "specular_mean": float(np.mean(s)),
            "visibility_mean": float(np.mean(v)),
        })

    weighted_vis = (vis_weighted_acc / np.maximum(vis_weight_sum, 1e-6)).astype(np.float32) if rig.lights else visibility

    env = rig.environment
    env_active = env is not None and (
        float(env.strength) > 0.0
        or float(env.key_strength) > 0.0
        or (env.env_map is not None and float(env.env_map_strength) > 0.0)
    )
    if env_active and env is not None:
        anisotropy, anisotropy_angle = _effective_anisotropy_maps(scene)
        view = _normalize(-points)
        env_d, env_s = evaluate_environment_lighting(
            normals, view, roughness, metalness, spec_level, specular_color, env,
            anisotropy=anisotropy, anisotropy_angle=anisotropy_angle,
            clearcoat=clearcoat, clearcoat_roughness=clearcoat_roughness,
        )
        # Prevent double-counting diffuse ambient fill when HDRI / dome irradiance is active.
        env_diff_cov = min(
            1.0,
            (float(env.strength) + (float(env.env_map_strength) if env.env_map is not None else 0.0))
            * float(env.diffuse_strength),
        )
        diffuse = (ambient_term * (1.0 - env_diff_cov) + direct_diffuse + env_d * eff_ao).astype(np.float32)
        specular += env_s
    else:
        diffuse = (ambient_term + direct_diffuse).astype(np.float32)
    if return_details:
        return diffuse, specular, visibility, {
            "per_light_contributions": per_light_records,
            "per_light_visibility": per_light_vis,
            "shadow_visibility_weighted": weighted_vis,
        }
    return diffuse, specular, visibility
