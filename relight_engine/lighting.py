from __future__ import annotations
import math
import cv2
import numpy as np
from .color import srgb_to_linear
from .contracts import CameraModel, EnvironmentLight, Light, LightingRig, SceneBuffers, EngineSettings
from .shadows import scene_shadow_visibility

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)

def _area_samples(light: Light, target_point: np.ndarray | None = None) -> list[np.ndarray]:
    center = np.asarray(light.position, np.float32)
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
    n = int(math.ceil(math.sqrt(light.samples)))
    offsets = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    result = []
    for iy in range(n):
        for ix in range(n):
            if len(result) >= light.samples:
                break
            result.append(center + right * offsets[ix] * light.area_size[0] + up_axis * offsets[iy] * light.area_size[1])
    return result

def _effective_material_maps(
    scene: SceneBuffers,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve authored/estimated maps against explicit conservative renderer priors.

    A supplied material_confidence map blends supplied properties toward the priors.
    Without that map, supplied maps are applied as provided but their provenance remains
    the caller's responsibility. No material is inferred from its semantic label.
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
    confidence = (
        np.clip(np.asarray(scene.material_confidence, dtype=np.float32), 0.0, 1.0)
        if scene.material_confidence is not None else None
    )

    def resolve(name: str, supplied: np.ndarray | None, prior: float):
        if supplied is None:
            return np.full((h, w), prior, dtype=np.float32) if name != "specular_color" else np.full((h, w, 3), prior, dtype=np.float32)
        values = np.asarray(supplied, dtype=np.float32)
        if name == "specular_color":
            if values.ndim == 2:
                values = np.repeat(values[..., None], 3, axis=-1)
            elif values.ndim == 1 and values.shape[0] == 3:
                values = np.broadcast_to(values, (h, w, 3)).copy()
        if confidence is None:
            return values
        weight = confidence if values.ndim == 2 else confidence[..., None]
        base = prior if values.ndim == 2 else np.full((1, 1, 3), prior, dtype=np.float32)
        return (base + weight * (values - base)).astype(np.float32)

    return (
        resolve("roughness", scene.roughness, priors["roughness"]),
        resolve("metalness", scene.metalness, priors["metalness"]),
        resolve("specular_level", scene.specular_level, priors["specular_level"]),
        resolve("specular_color", scene.specular_color, priors["specular_color"]),
        resolve("clearcoat", scene.clearcoat, priors["clearcoat"]),
        resolve("clearcoat_roughness", scene.clearcoat_roughness, priors["clearcoat_roughness"]),
        resolve("subsurface", scene.subsurface, priors["subsurface"]),
    )


def _effective_anisotropy_maps(scene: SceneBuffers) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve optional anisotropy magnitude [0, 1] (confidence-gated) and tangent rotation angle."""
    if scene.anisotropy is None:
        return None, None
    aniso = np.clip(np.asarray(scene.anisotropy, dtype=np.float32), 0.0, 1.0)
    if scene.material_confidence is not None:
        conf = np.clip(np.asarray(scene.material_confidence, dtype=np.float32), 0.0, 1.0)
        aniso = (aniso * conf).astype(np.float32)
    angle = (
        np.asarray(scene.anisotropy_angle, dtype=np.float32)
        if scene.anisotropy_angle is not None
        else None
    )
    return aniso, angle


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
        coat_spec = _ggx_specular(
            normals, view, light_dir, clearcoat_roughness, zeros, f04, specular_color
        )
        spec = spec + clearcoat[..., None] * coat_spec
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
    """Build roughness-prefiltered specular mip chain and cosine-weighted diffuse irradiance map."""
    env_lin = np.maximum(np.asarray(env_map, dtype=np.float32), 0.0)
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
        env_specular = env_specular + clearcoat[..., None] * coat_spec

    env_specular = (env_specular * float(env.specular_strength)).astype(np.float32)
    env_diffuse = (irrad_total * np.clip(1.0 - 0.88 * metal, 0.08, 1.0) * float(env.diffuse_strength)).astype(np.float32)
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
    diffuse_energy = np.clip(1.0 - 0.88 * metalness, 0.08, 1.0).astype(np.float32)[..., None]
    has_coat = bool(np.any(clearcoat > 1e-4))
    has_sss = bool(np.any(subsurface > 1e-4))
    view = _normalize(-points)
    diffuse = np.zeros((h, w, 3), np.float32)
    specular = np.zeros((h, w, 3), np.float32)
    visibility_sum = np.zeros((h, w), np.float32)

    if light.kind == 'directional':
        toward = np.asarray(light.direction, np.float32)
        toward /= max(float(np.linalg.norm(toward)), 1e-8)
        dirs = [np.broadcast_to(toward, (h, w, 3))]
        positions = [None]
    elif light.kind == 'point':
        lp = np.asarray(light.position, np.float32)
        vec = lp[None, None, :] - points
        dist = np.linalg.norm(vec, axis=-1)
        dirs = [vec / np.maximum(dist[..., None], 1e-8)]
        positions = [np.broadcast_to(lp, (h, w, 3))]
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
            attenuation = np.full((h, w), light.strength, np.float32)
        elif scene.depth_kind == 'metric':
            attenuation = light.strength / np.maximum(dist * dist, 0.04)
        else:
            # Relative depth has unknown scale: use a stable normalized falloff, not fake inverse-square metres.
            attenuation = light.strength / (1.0 + 0.15 * dist * dist)
        if light.casts_shadows:
            visibility = scene_shadow_visibility(
                scene, camera, points, normals, pos,
                d if light.kind == 'directional' else None,
                settings, light_radius=float(light.radius),
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
    """1-bounce screen-space indirect inter-reflection (SSGI color bleed) between surfaces."""
    strength = float(settings.indirect_bounce_strength)
    if strength <= 0.0:
        return np.zeros_like(direct_diffuse, dtype=np.float32)

    h, w = scene.depth.shape
    albedo_lin = (
        srgb_to_linear(scene.albedo)
        if scene.albedo is not None
        else np.clip(srgb_to_linear(scene.rgb_srgb), 0.02, 0.85)
    )
    radiance = (albedo_lin * np.maximum(direct_diffuse, 0.0)).astype(np.float32)
    valid = scene.depth > 0
    log_d = np.log(np.maximum(scene.depth, 1e-4)).astype(np.float32)
    z_med = max(float(np.median(scene.depth[valid])) if np.any(valid) else 1.0, 1e-3)
    world_scale = z_med * 0.35

    r_max = max(2, int(round(settings.indirect_bounce_radius)))
    r_mid = max(1, r_max // 2)
    offsets = []
    for rad in (r_mid, r_max):
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
        log_s = np.roll(np.roll(log_d, dy, axis=0), dx, axis=1)
        val_s = np.roll(np.roll(valid, dy, axis=0), dx, axis=1).copy()
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
        depth_weight = np.exp(-np.abs(log_s - log_d) / 0.25)
        dist_weight = 1.0 / (1.0 + dist2 / max(world_scale * world_scale, 1e-4))
        weight = np.where(valid & val_s, facing * depth_weight * dist_weight, 0.0).astype(np.float32)
        accum += rad_s * weight[..., None]
        w_sum += weight

    indirect = accum / np.maximum(w_sum[..., None], 1.0)
    return (indirect * strength).astype(np.float32)

def evaluate_rig(scene: SceneBuffers, camera: CameraModel, points: np.ndarray,
                 normals: np.ndarray, rig: LightingRig, settings: EngineSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rig.validate()
    h, w = scene.depth.shape
    roughness, metalness, spec_level, specular_color, clearcoat, clearcoat_roughness, _ = _effective_material_maps(scene)
    diffuse_energy = np.clip(1.0 - 0.88 * metalness, 0.08, 1.0).astype(np.float32)[..., None]
    sky_color = np.asarray(rig.ambient_color, np.float32)[None, None, :]
    if rig.ambient_ground_color is not None:
        ground_color = np.asarray(rig.ambient_ground_color, np.float32)[None, None, :]
        sky_t = np.clip(0.5 * (normals[..., 1] + 1.0), 0.0, 1.0)[..., None]
        ambient = (ground_color * (1.0 - sky_t) + sky_color * sky_t) * rig.ambient_strength
    else:
        ambient = sky_color * rig.ambient_strength
    ambient_term = (np.broadcast_to(ambient, (h, w, 3)).astype(np.float32, copy=True) * diffuse_energy).astype(np.float32)

    direct_diffuse = np.zeros((h, w, 3), np.float32)
    specular = np.zeros((h, w, 3), np.float32)
    visibility = np.ones((h, w), np.float32)
    for light in rig.lights:
        d, s, v = _light_contribution(scene, camera, points, normals, light, settings)
        direct_diffuse += d
        specular += s
        visibility = np.minimum(visibility, v)

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
        diffuse = (ambient_term * (1.0 - env_diff_cov) + direct_diffuse + env_d).astype(np.float32)
        specular += env_s
    else:
        diffuse = (ambient_term + direct_diffuse).astype(np.float32)
    return diffuse, specular, visibility
