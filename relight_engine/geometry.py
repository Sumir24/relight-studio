from __future__ import annotations
import numpy as np
from .contracts import CameraModel, SceneBuffers

def backproject_depth(depth: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Back-project positive camera-Z depth into camera coordinates (+X right, +Y up, +Z forward)."""
    camera.validate()
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth.astype(np.float32)
    x = (xx - camera.cx) * z / camera.fx
    y = (camera.cy - yy) * z / camera.fy
    return np.stack([x, y, z], axis=-1)

def estimate_normals(points: np.ndarray, depth: np.ndarray, depth_confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate camera-space normals from a depth surface; normals face the camera.

    This is a 2.5D height-field normal estimate. It cannot recover unseen backsides,
    overhangs, or geometry hidden behind the first visible surface.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError('points must be HxWx3.')
    # Centered finite differences are less biased than forward-only gradients.
    gy = np.gradient(points, axis=0)
    gx = np.gradient(points, axis=1)
    n = np.cross(gx, gy)
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(length, 1e-8)
    # With image Y increasing downward, cross(dP/dx, dP/dy) faces the camera for a fronto-parallel plane.
    view_to_camera = -points / np.maximum(np.linalg.norm(points, axis=-1, keepdims=True), 1e-8)
    flip = np.sum(n * view_to_camera, axis=-1, keepdims=True) < 0
    n = np.where(flip, -n, n)
    valid = (depth > 0).astype(np.float32)
    # Reduce confidence around abrupt depth discontinuities where finite differences mix surfaces.
    z = np.maximum(depth.astype(np.float32), 1e-6)
    dzx = np.abs(np.gradient(z, axis=1)) / z
    dzy = np.abs(np.gradient(z, axis=0)) / z
    edge_conf = np.exp(-np.maximum(dzx, dzy) * 8.0)
    conf = np.clip(depth_confidence.astype(np.float32), 0, 1) * valid * edge_conf
    conf *= np.clip(length[..., 0] / 1e-6, 0, 1)
    n[valid < 0.5] = np.array([0, 0, -1], np.float32)
    return n.astype(np.float32), conf.astype(np.float32)

def prepare_geometry(scene: SceneBuffers, camera: CameraModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = backproject_depth(scene.depth, camera)
    if scene.normals is not None:
        normals = scene.normals.astype(np.float32)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
        gconf = scene.geometry_confidence if scene.geometry_confidence is not None else scene.depth_confidence
    else:
        normals, gconf = estimate_normals(points, scene.depth, scene.depth_confidence)
    return points, normals, np.clip(gconf.astype(np.float32), 0, 1)
