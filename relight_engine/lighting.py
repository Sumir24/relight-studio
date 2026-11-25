from __future__ import annotations
import math
import numpy as np
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

def _effective_material_maps(scene: SceneBuffers) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    roughness, metalness, spec_level, specular_color = _effective_material_maps(scene)
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
            visibility = scene_shadow_visibility(scene, camera, points, normals, pos,
                                                 d if light.kind == 'directional' else None, settings)
        else:
            visibility = np.ones((h, w), np.float32)
        direct = (ndotl_diffuse * attenuation * visibility / count).astype(np.float32)
        diffuse += direct[..., None] * color
        spec = _ggx_specular(
            normals, view, d, roughness, metalness, spec_level, specular_color
        )
        specular += spec * (attenuation * visibility / count)[..., None] * color
        visibility_sum += visibility / count
    return diffuse, specular, visibility_sum

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
