from __future__ import annotations

import numpy as np

from .contracts import CameraModel, SceneBuffers


def backproject_depth(depth: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Back-project positive camera-Z depth into camera coordinates.

    Coordinates are +X right, +Y up and +Z forward/away from the camera.
    Relative depth remains relative: the returned point coordinates are not metres.
    """
    camera.validate()
    z = np.asarray(depth, dtype=np.float32)
    if z.ndim != 2 or min(z.shape, default=0) < 2:
        raise ValueError("depth must be a 2D array at least 2x2 pixels.")
    if not np.isfinite(z).all() or np.any(z < 0):
        raise ValueError("depth must contain finite, non-negative values; zero denotes invalid depth.")

    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx - camera.cx) * z / camera.fx
    y = (camera.cy - yy) * z / camera.fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _shifted(array: np.ndarray, offset: int, axis: int) -> np.ndarray:
    """Shift an array without allocating a padded copy; wrapped borders are masked."""
    return np.roll(array, offset, axis=axis)


def _axis_derivative(
    points: np.ndarray,
    log_depth: np.ndarray,
    valid: np.ndarray,
    axis: int,
    relative_edge_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a +axis tangent without differencing across a strong depth edge.

    Central differences are used in locally smooth regions. At a discontinuity,
    the one-sided tangent from the neighbour with the closest log-depth is used.
    The second return value indicates whether a valid tangent could be formed.
    """
    minus_points = _shifted(points, 1, axis)
    plus_points = _shifted(points, -1, axis)
    minus_log = _shifted(log_depth, 1, axis)
    plus_log = _shifted(log_depth, -1, axis)
    minus_valid = _shifted(valid, 1, axis).copy()
    plus_valid = _shifted(valid, -1, axis).copy()

    if axis == 1:
        minus_valid[:, 0] = False
        plus_valid[:, -1] = False
    elif axis == 0:
        minus_valid[0, :] = False
        plus_valid[-1, :] = False
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns).")

    delta_minus = np.abs(log_depth - minus_log)
    delta_plus = np.abs(log_depth - plus_log)
    both = valid & minus_valid & plus_valid
    central_ok = both & (np.maximum(delta_minus, delta_plus) <= relative_edge_threshold)
    choose_minus = minus_valid & (~plus_valid | (delta_minus <= delta_plus))

    central = (plus_points - minus_points) * 0.5
    one_sided_minus = points - minus_points
    one_sided_plus = plus_points - points
    derivative = np.where(
        central_ok[..., None],
        central,
        np.where(choose_minus[..., None], one_sided_minus, one_sided_plus),
    )
    has_neighbour = valid & (minus_valid | plus_valid)
    derivative[~has_neighbour] = 0.0
    return derivative.astype(np.float32), has_neighbour


def _log_depth_curvature(log_depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return a second-difference edge/curvature signal in log-depth space."""
    curvature = np.zeros(log_depth.shape, dtype=np.float32)
    for axis in (0, 1):
        minus = _shifted(log_depth, 1, axis)
        plus = _shifted(log_depth, -1, axis)
        minus_valid = _shifted(valid, 1, axis).copy()
        plus_valid = _shifted(valid, -1, axis).copy()
        if axis == 0:
            minus_valid[0, :] = False
            plus_valid[-1, :] = False
        else:
            minus_valid[:, 0] = False
            plus_valid[:, -1] = False
        supported = valid & minus_valid & plus_valid
        second_difference = np.abs(plus + minus - 2.0 * log_depth)
        curvature = np.maximum(curvature, np.where(supported, second_difference, 0.0))
    return curvature


def _neighbour_support_fraction(valid: np.ndarray) -> np.ndarray:
    """Measure the fraction of in-image 4-neighbour samples that have valid depth."""
    left = _shifted(valid, 1, axis=1).copy()
    right = _shifted(valid, -1, axis=1).copy()
    up = _shifted(valid, 1, axis=0).copy()
    down = _shifted(valid, -1, axis=0).copy()
    left[:, 0] = False
    right[:, -1] = False
    up[0, :] = False
    down[-1, :] = False

    observed = (
        left.astype(np.float32)
        + right.astype(np.float32)
        + up.astype(np.float32)
        + down.astype(np.float32)
    )
    expected = np.full(valid.shape, 4.0, dtype=np.float32)
    expected[:, 0] -= 1.0
    expected[:, -1] -= 1.0
    expected[0, :] -= 1.0
    expected[-1, :] -= 1.0
    return np.divide(observed, expected, out=np.zeros_like(observed), where=expected > 0)


def estimate_normals(
    points: np.ndarray,
    depth: np.ndarray,
    depth_confidence: np.ndarray,
    *,
    relative_edge_threshold: float = 0.08,
    edge_decay: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate camera-space visible-surface normals and a heuristic reliability map.

    Tangents use central differences in smooth regions and one-sided differences
    near depth discontinuities, reducing cross-surface normal contamination.

    The returned confidence is NOT calibrated model uncertainty. It is the input
    depth-confidence prior multiplied by local depth-edge consistency and normal
    validity. Relative-depth inputs produce relative 2.5D geometry, not metric 3D.
    """
    points = np.asarray(points, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32)
    depth_confidence = np.asarray(depth_confidence, dtype=np.float32)

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape HxWx3.")
    if min(points.shape[:2], default=0) < 2:
        raise ValueError("points must be at least 2x2 pixels.")
    if depth.shape != points.shape[:2]:
        raise ValueError("depth must match points spatial dimensions.")
    if depth_confidence.shape != depth.shape:
        raise ValueError("depth_confidence must match depth dimensions.")
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("depth must be finite and non-negative.")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values.")
    if not np.isfinite(depth_confidence).all() or np.any(depth_confidence < 0) or np.any(depth_confidence > 1):
        raise ValueError("depth_confidence must be finite and in [0,1].")
    if not (np.isfinite(relative_edge_threshold) and relative_edge_threshold >= 0):
        raise ValueError("relative_edge_threshold must be finite and non-negative.")
    if not (np.isfinite(edge_decay) and edge_decay >= 0):
        raise ValueError("edge_decay must be finite and non-negative.")

    valid = (depth > 0) & np.isfinite(points).all(axis=-1)
    log_depth = np.log(np.maximum(depth, 1e-6))

    gx, support_x = _axis_derivative(
        points, log_depth, valid, axis=1,
        relative_edge_threshold=relative_edge_threshold,
    )
    gy, support_y = _axis_derivative(
        points, log_depth, valid, axis=0,
        relative_edge_threshold=relative_edge_threshold,
    )

    raw_normals = np.cross(gx, gy)
    normal_length = np.linalg.norm(raw_normals, axis=-1, keepdims=True)
    normal_valid = (
        valid
        & support_x
        & support_y
        & (normal_length[..., 0] > 1e-8)
    )
    normals = raw_normals / np.maximum(normal_length, 1e-8)

    view_to_camera = -points / np.maximum(
        np.linalg.norm(points, axis=-1, keepdims=True), 1e-8
    )
    flip = np.sum(normals * view_to_camera, axis=-1, keepdims=True) < 0
    normals = np.where(flip, -normals, normals)

    curvature = _log_depth_curvature(log_depth, valid)
    edge_excess = np.maximum(curvature - relative_edge_threshold, 0.0)
    edge_consistency = np.exp(-edge_excess * edge_decay)

    # This is a reliability heuristic, not a probability or calibrated uncertainty.
    neighbour_support = _neighbour_support_fraction(valid)
    confidence = (
        np.clip(depth_confidence, 0.0, 1.0)
        * edge_consistency
        * neighbour_support
        * normal_valid.astype(np.float32)
    )

    fallback_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    normals[~normal_valid] = fallback_normal
    return normals.astype(np.float32), np.clip(confidence, 0.0, 1.0).astype(np.float32)


def refine_depth_to_rgb_edges(
    depth: np.ndarray,
    rgb_srgb: np.ndarray,
    *,
    sigma: float = 3.5,
    eps_guide: float = 0.004,
) -> np.ndarray:
    """Snap soft relative depth transitions onto sharp RGB luminance edges via guided filtering."""
    from scipy.ndimage import gaussian_filter

    d = np.asarray(depth, dtype=np.float32)
    valid = d > 0
    if not np.any(valid):
        return d
    guide = (
        0.299 * rgb_srgb[..., 0] + 0.587 * rgb_srgb[..., 1] + 0.114 * rgb_srgb[..., 2]
    ).astype(np.float32)
    mean_i = gaussian_filter(guide, sigma, mode='reflect')
    mean_ii = gaussian_filter(guide * guide, sigma, mode='reflect')
    var_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)
    mean_p = gaussian_filter(d, sigma, mode='reflect')
    mean_ip = gaussian_filter(guide * d, sigma, mode='reflect')
    cov_ip = mean_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps_guide)
    b = mean_p - a * mean_i
    refined = gaussian_filter(a, sigma, mode='reflect') * guide + gaussian_filter(b, sigma, mode='reflect')
    d_min = float(np.min(d[valid]))
    d_max = float(np.max(d[valid]))
    refined = np.clip(refined, d_min, d_max)
    return np.where(valid, refined, 0.0).astype(np.float32)


def prepare_geometry(
    scene: SceneBuffers,
    camera: CameraModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare points, normalized normals and a [0,1] geometry-reliability map."""
    points = backproject_depth(scene.depth, camera)
    valid_depth = scene.depth > 0

    if scene.normals is not None:
        normals = np.array(scene.normals, dtype=np.float32, copy=True)
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals /= np.maximum(lengths, 1e-8)
        normals[~valid_depth] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        if scene.geometry_confidence is not None:
            geometry_confidence = np.asarray(scene.geometry_confidence, dtype=np.float32).copy()
        else:
            geometry_confidence = np.asarray(scene.depth_confidence, dtype=np.float32).copy()
    else:
        normals, geometry_confidence = estimate_normals(
            points, scene.depth, scene.depth_confidence
        )

    geometry_confidence = np.clip(geometry_confidence, 0.0, 1.0)
    geometry_confidence[~valid_depth] = 0.0
    return points, normals.astype(np.float32), geometry_confidence.astype(np.float32)
