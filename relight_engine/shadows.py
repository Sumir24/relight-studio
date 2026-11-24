from __future__ import annotations
import numpy as np
import cv2
from .contracts import CameraModel, SceneBuffers, EngineSettings

def _smooth_visibility_edge_aware(visibility: np.ndarray, depth: np.ndarray,
                                 edge_threshold: float = 0.08) -> np.ndarray:
    """Smooth step-dithered visibility across continuous surfaces without bleeding across depth steps."""
    valid = depth > 0
    log_d = np.log(np.maximum(depth, 1e-6))
    acc = visibility.astype(np.float32).copy()
    wsum = np.ones_like(acc)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        vis_n = np.roll(np.roll(visibility, dy, axis=0), dx, axis=1)
        log_n = np.roll(np.roll(log_d, dy, axis=0), dx, axis=1)
        val_n = np.roll(np.roll(valid, dy, axis=0), dx, axis=1).copy()
        if dy == 1:
            val_n[0, :] = False
        elif dy == -1:
            val_n[-1, :] = False
        if dx == 1:
            val_n[:, 0] = False
        elif dx == -1:
            val_n[:, -1] = False
        spatial_w = 1.0 if (dy == 0 or dx == 0) else 0.7071
        same_surface = valid & val_n & (np.abs(log_d - log_n) <= edge_threshold)
        w = np.where(same_surface, spatial_w, 0.0).astype(np.float32)
        acc += w * vis_n
        wsum += w
    return np.clip(acc / np.maximum(wsum, 1e-6), 0.0, 1.0).astype(np.float32)


def ray_visibility(points: np.ndarray, normals: np.ndarray, depth: np.ndarray,
                   camera: CameraModel, to_light: np.ndarray,
                   max_distance: np.ndarray, depth_kind: str,
                   settings: EngineSettings) -> np.ndarray:
    """Approximate 2.5D visibility by marching camera-projected rays through the depth map.

    A nearer observed depth than the ray point marks an occluder. Uses deterministic
    2x2 Bayer sub-step phase offsets, soft depth-margin attenuation, and edge-aware
    spatial smoothing to avoid discrete step-band silhouettes.
    """
    h, w = depth.shape
    direction = to_light / np.maximum(np.linalg.norm(to_light, axis=-1, keepdims=True), 1e-8)
    bias = settings.shadow_bias_metric if depth_kind == 'metric' else settings.shadow_bias_relative
    origin = points + normals * bias * 2.0
    visibility = np.ones((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    bayer = (((yy & 1) * 2 + (xx & 1)) * 0.25).astype(np.float32)
    n_steps = max(2, settings.shadow_steps)
    step_base = np.linspace(0.06, 0.96, n_steps, dtype=np.float32)
    delta_u = (0.96 - 0.06) / max(n_steps - 1, 1)
    soft_width = max(bias * 3.0, 1e-4)
    depth_f32 = depth.astype(np.float32)
    for base_u in step_base:
        u_step = np.clip(base_u + (bayer - 0.375) * (delta_u * 0.85), 0.02, 0.98)
        step = u_step ** 1.8
        q = origin + direction * (max_distance * step)[..., None]
        qz = q[..., 2]
        safe_z = np.maximum(qz, 1e-5)
        u = camera.fx * q[..., 0] / safe_z + camera.cx
        v = camera.cy - camera.fy * q[..., 1] / safe_z
        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)
        sampled = cv2.remap(depth_f32, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inside = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1) & (qz > 0)
        penetration = (qz - bias) - sampled
        soft_hit = np.where(
            inside & (sampled > 0) & (penetration > 0),
            np.clip(penetration / soft_width, 0.0, 1.0),
            0.0,
        )
        visibility = np.minimum(visibility, 1.0 - soft_hit)
        if not np.any(visibility > 0.02):
            break
    return _smooth_visibility_edge_aware(visibility, depth_f32)

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
    bias = settings.shadow_bias_metric if scene.depth_kind == 'metric' else settings.shadow_bias_relative
    valid = scene.depth > 0
    if np.any(valid):
        valid_points = points[valid]
        scene_diag = max(float(np.linalg.norm(np.max(valid_points, axis=0) - np.min(valid_points, axis=0))), 1.0)
        z_min = float(np.min(scene.depth[valid]))
        z_max = float(np.max(scene.depth[valid]))
    else:
        scene_diag = 1.0
        z_min, z_max = 0.0, 1.0
    direction = directional_vectors / np.maximum(np.linalg.norm(directional_vectors, axis=-1, keepdims=True), 1e-8)
    dz = direction[..., 2]
    max_dist = np.full((h, w), scene_diag, np.float32)
    toward_cam = dz < -1e-4
    away_cam = dz > 1e-4
    if np.any(toward_cam):
        t_exit_near = np.maximum((points[..., 2] - z_min) / np.maximum(-dz, 1e-4), 0.0) + 4.0 * bias
        max_dist = np.where(toward_cam, np.minimum(max_dist, t_exit_near), max_dist)
    if np.any(away_cam):
        t_exit_far = np.maximum((z_max - points[..., 2]) / np.maximum(dz, 1e-4), 0.0) + 4.0 * bias
        max_dist = np.where(away_cam, np.minimum(max_dist, t_exit_far), max_dist)
    max_dist = np.maximum(max_dist, 4.0 * bias)
    return ray_visibility(points, normals, scene.depth, camera, directional_vectors, max_dist,
                          scene.depth_kind, settings)
