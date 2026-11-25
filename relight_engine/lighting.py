from __future__ import annotations
import math
import numpy as np
from .color import srgb_to_linear
from .contracts import CameraModel, Light, LightingRig, SceneBuffers, EngineSettings
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
            diffuse += ndotl_rgb * ((attenuation * visibility / count)[..., None] * color)
        else:
            direct = (ndotl_diffuse * attenuation * visibility / count).astype(np.float32)
            diffuse += direct[..., None] * color
        spec = _ggx_specular(
            normals, view, d, roughness, metalness, spec_level, specular_color
        )
        if has_coat:
            zeros = np.zeros((h, w), dtype=np.float32)
            f04 = np.full((h, w), 0.04, dtype=np.float32)
            coat_spec = _ggx_specular(
                normals, view, d, clearcoat_roughness, zeros, f04, specular_color
            )
            spec = spec + clearcoat[..., None] * coat_spec
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
    sky_color = np.asarray(rig.ambient_color, np.float32)[None, None, :]
    if rig.ambient_ground_color is not None:
        ground_color = np.asarray(rig.ambient_ground_color, np.float32)[None, None, :]
        sky_t = np.clip(0.5 * (normals[..., 1] + 1.0), 0.0, 1.0)[..., None]
        ambient = (ground_color * (1.0 - sky_t) + sky_color * sky_t) * rig.ambient_strength
    else:
        ambient = sky_color * rig.ambient_strength
    diffuse = np.broadcast_to(ambient, (h, w, 3)).astype(np.float32, copy=True)
    specular = np.zeros((h, w, 3), np.float32)
    visibility = np.ones((h, w), np.float32)
    for light in rig.lights:
        d, s, v = _light_contribution(scene, camera, points, normals, light, settings)
        diffuse += d
        specular += s
        visibility = np.minimum(visibility, v)
    return diffuse, specular, visibility
