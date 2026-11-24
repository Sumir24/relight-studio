from __future__ import annotations
import numpy as np
import cv2
from .contracts import CameraModel, SceneBuffers, EngineSettings

def ray_visibility(points: np.ndarray, normals: np.ndarray, depth: np.ndarray,
                   camera: CameraModel, to_light: np.ndarray,
                   max_distance: np.ndarray, depth_kind: str,
                   settings: EngineSettings) -> np.ndarray:
    """Approximate 2.5D visibility by marching camera-projected rays through the depth map.

    A nearer observed depth than the ray point marks an occluder. This is not a full
    mesh shadow map: it cannot represent unseen/hidden geometry or true overhangs.
    """
    h, w = depth.shape
    direction = to_light / np.maximum(np.linalg.norm(to_light, axis=-1, keepdims=True), 1e-8)
    bias = settings.shadow_bias_metric if depth_kind == 'metric' else settings.shadow_bias_relative
    origin = points + normals * bias * 2.0
    visibility = np.ones((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for step in np.linspace(0.10, 0.92, max(2, settings.shadow_steps), dtype=np.float32):
        q = origin + direction * (max_distance[..., None] * step)
        qz = q[..., 2]
        safe_z = np.maximum(qz, 1e-5)
        u = camera.fx * q[..., 0] / safe_z + camera.cx
        v = camera.cy - camera.fy * q[..., 1] / safe_z
        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)
        sampled = cv2.remap(depth.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inside = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1) & (qz > 0)
        occluded = inside & (sampled > 0) & (sampled < (qz - bias))
        visibility[occluded] = 0.0
        if not np.any(visibility > 0):
            break
    return visibility

def scene_shadow_visibility(scene: SceneBuffers, camera: CameraModel, points: np.ndarray,
                            normals: np.ndarray, light_positions: np.ndarray | None,
                            directional_vectors: np.ndarray | None,
                            settings: EngineSettings) -> np.ndarray:
    h, w = scene.depth.shape
    if light_positions is not None:
        vec = light_positions - points
        dist = np.linalg.norm(vec, axis=-1)
        return ray_visibility(points, normals, scene.depth, camera, vec, dist, scene.depth_kind, settings)
    assert directional_vectors is not None
    max_dist = np.full((h, w), max(float(np.max(scene.depth)), 1.0) * 2.0, np.float32)
    return ray_visibility(points, normals, scene.depth, camera, directional_vectors, max_dist,
                          scene.depth_kind, settings)
