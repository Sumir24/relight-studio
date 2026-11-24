from __future__ import annotations
import math
import numpy as np
from .contracts import CameraModel, Light, LightingRig, SceneBuffers, EngineSettings
from .shadows import scene_shadow_visibility

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)

def _area_samples(light: Light) -> list[np.ndarray]:
    center = np.asarray(light.position, np.float32)
    toward_scene = np.array([0, 0, 1], np.float32) - center
    toward_scene /= max(float(np.linalg.norm(toward_scene)), 1e-8)
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

def _ggx_specular(normals: np.ndarray, view: np.ndarray, light_dir: np.ndarray,
                  roughness: np.ndarray, metalness: np.ndarray,
                  specular_level: np.ndarray) -> np.ndarray:
    ndotl = np.clip(np.sum(normals * light_dir, axis=-1), 0, 1)
    ndotv = np.clip(np.sum(normals * view, axis=-1), 1e-4, 1)
    half_vec = _normalize(light_dir + view)
    ndoth = np.clip(np.sum(normals * half_vec, axis=-1), 0, 1)
    vdoth = np.clip(np.sum(view * half_vec, axis=-1), 0, 1)
    alpha = np.maximum(roughness, 0.045) ** 2
    alpha2 = alpha ** 2
    denom = np.maximum(np.pi * (ndoth * ndoth * (alpha2 - 1) + 1) ** 2, 1e-7)
    D = alpha2 / denom
    k = (alpha + 1) ** 2 / 8.0
    Gv = ndotv / (ndotv * (1 - k) + k)
    Gl = ndotl / (ndotl * (1 - k) + k)
    G = Gv * Gl
    f0 = specular_level * (1 - metalness) + metalness * 0.65
    F = f0 + (1 - f0) * (1 - vdoth) ** 5
    spec = (D * G * F) / np.maximum(4 * ndotv * ndotl, 1e-5)
    return (spec * ndotl).astype(np.float32)

def _light_contribution(scene: SceneBuffers, camera: CameraModel, points: np.ndarray,
                        normals: np.ndarray, light: Light, settings: EngineSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = scene.depth.shape
    color = np.asarray(light.color, np.float32)[None, None, :]
    roughness = scene.roughness if scene.roughness is not None else np.full((h, w), 0.75, np.float32)
    metalness = scene.metalness if scene.metalness is not None else np.zeros((h, w), np.float32)
    spec_level = scene.specular_level if scene.specular_level is not None else np.full((h, w), 0.04, np.float32)
    view = _normalize(-points)
    diffuse = np.zeros((h, w, 3), np.float32)
    specular = np.zeros((h, w, 3), np.float32)
    visibility_sum = np.zeros((h, w), np.float32)

    if light.kind == 'directional':
        toward = np.asarray(light.direction, np.float32)
        toward /= max(float(np.linalg.norm(toward)), 1e-8)
        dirs = [np.broadcast_to(toward, (h, w, 3))]
        positions = [None]
        max_dists = [np.full((h, w), max(float(np.max(scene.depth)), 1.0) * 2.0, np.float32)]
    elif light.kind == 'point':
        lp = np.asarray(light.position, np.float32)
        vec = lp[None, None, :] - points
        dist = np.linalg.norm(vec, axis=-1)
        dirs = [vec / np.maximum(dist[..., None], 1e-8)]
        positions = [np.broadcast_to(lp, (h, w, 3))]
        max_dists = [dist]
    else:
        positions = [np.broadcast_to(p, (h, w, 3)) for p in _area_samples(light)]
        dirs, max_dists = [], []
        for pos in positions:
            vec = pos - points
            dist = np.linalg.norm(vec, axis=-1)
            dirs.append(vec / np.maximum(dist[..., None], 1e-8))
            max_dists.append(dist)

    count = len(dirs)
    for d, pos, dist in zip(dirs, positions, max_dists):
        ndotl = np.clip(np.sum(normals * d, axis=-1), 0, 1)
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
        direct = (ndotl * attenuation * visibility / count).astype(np.float32)
        diffuse += direct[..., None] * color
        spec = _ggx_specular(normals, view, d, roughness, metalness, spec_level)
        specular += (spec * attenuation * visibility / count)[..., None] * color
        visibility_sum += visibility / count
    return diffuse, specular, visibility_sum

def evaluate_rig(scene: SceneBuffers, camera: CameraModel, points: np.ndarray,
                 normals: np.ndarray, rig: LightingRig, settings: EngineSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rig.validate()
    h, w = scene.depth.shape
    ambient = np.asarray(rig.ambient_color, np.float32)[None, None, :] * rig.ambient_strength
    diffuse = np.broadcast_to(ambient, (h, w, 3)).copy()
    specular = np.zeros((h, w, 3), np.float32)
    visibility = np.ones((h, w), np.float32)
    vis_weight = 0
    for light in rig.lights:
        d, s, v = _light_contribution(scene, camera, points, normals, light, settings)
        diffuse += d
        specular += s
        visibility = np.minimum(visibility, v)
        vis_weight += 1
    return diffuse, specular, visibility
