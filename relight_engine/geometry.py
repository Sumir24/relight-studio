from __future__ import annotations

from collections import OrderedDict
import hashlib
import threading
import numpy as np

from .contracts import CameraModel, SceneBuffers

_GEOMETRY_CACHE: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
_AO_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_GEOMETRY_CACHE_LOCK = threading.Lock()
_GEOMETRY_CACHE_MAX = 4


def _geom_digest(arr: np.ndarray | None) -> tuple | None:
    if arr is None:
        return None
    buf = np.ascontiguousarray(arr)
    return (buf.shape, str(buf.dtype), hashlib.blake2b(memoryview(buf), digest_size=16).digest())


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
    """Multi-scale RGB-guided edge snap tightening blurry depth silhouettes onto crisp RGB edges."""
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

    d = np.asarray(depth, dtype=np.float32)
    valid = d > 0
    if not np.any(valid):
        return d
    guide = (
        0.299 * rgb_srgb[..., 0] + 0.587 * rgb_srgb[..., 1] + 0.114 * rgb_srgb[..., 2]
    ).astype(np.float32)

    def _guided_pass(src_d: np.ndarray, sig: float) -> np.ndarray:
        mean_i = gaussian_filter(guide, sig, mode='reflect')
        mean_ii = gaussian_filter(guide * guide, sig, mode='reflect')
        var_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)
        mean_p = gaussian_filter(src_d, sig, mode='reflect')
        mean_pp = gaussian_filter(src_d * src_d, sig, mode='reflect')
        var_p = np.maximum(mean_pp - mean_p * mean_p, 0.0)
        mean_ip = gaussian_filter(guide * src_d, sig, mode='reflect')
        cov_ip = mean_ip - mean_i * mean_p
        a = cov_ip / (var_i + eps_guide)
        d_scale = max(float(np.median(src_d[valid])), 1e-3)
        geom_edge_support = np.clip(np.sqrt(var_p) / (0.015 * d_scale), 0.10, 1.0)
        a = a * geom_edge_support
        b = mean_p - a * mean_i
        return gaussian_filter(a, sig, mode='reflect') * guide + gaussian_filter(b, sig, mode='reflect')

    coarse = _guided_pass(d, float(sigma))
    fine = _guided_pass(coarse, max(1.2, float(sigma) * 0.4))
    refined = 0.45 * coarse + 0.55 * fine
    # Clamp strictly to local valid neighborhood bounds so guided filtering never overshoots.
    loc_min = minimum_filter(np.where(valid, d, np.inf), size=7)
    loc_max = maximum_filter(np.where(valid, d, 0.0), size=7)
    loc_min = np.where(np.isfinite(loc_min), loc_min, float(np.min(d[valid])))
    refined = np.clip(refined, loc_min, np.maximum(loc_max, loc_min))
    return np.where(valid, refined, 0.0).astype(np.float32)


def compute_depth_normal_consistency(
    points: np.ndarray,
    depth: np.ndarray,
    normals: np.ndarray,
) -> dict[str, object]:
    """Compare supplied surface normals against edge-aware depth-derived normals."""
    valid = depth > 0
    n_depth, depth_rel = estimate_normals(points, depth, np.ones_like(depth, dtype=np.float32))
    n_sup = np.asarray(normals, dtype=np.float32)
    n_sup = n_sup / np.maximum(np.linalg.norm(n_sup, axis=-1, keepdims=True), 1e-8)
    cos_sim = np.clip(np.sum(n_sup * n_depth, axis=-1), -1.0, 1.0).astype(np.float32)
    support = valid & (depth_rel > 0.25)
    if not np.any(support):
        support = valid
    angles = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))).astype(np.float32)
    disagreement = np.clip(0.5 * (1.0 - cos_sim), 0.0, 1.0).astype(np.float32)
    return {
        "cos_similarity": cos_sim,
        "disagreement": disagreement,
        "mean_angle_deg": float(np.mean(angles[support])) if np.any(support) else 0.0,
        "p95_angle_deg": float(np.percentile(angles[support], 95)) if np.any(support) else 0.0,
        "disagreement_fraction": float(np.mean(angles[support] > 35.0)) if np.any(support) else 0.0,
    }


def prepare_geometry(
    scene: SceneBuffers,
    camera: CameraModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare points, normalized normals and a [0,1] geometry-reliability map (content-keyed LRU cached)."""
    key = (
        _geom_digest(scene.depth),
        _geom_digest(scene.depth_confidence),
        _geom_digest(scene.normals),
        _geom_digest(scene.geometry_confidence),
        float(camera.fx),
        float(camera.fy),
        float(camera.cx),
        float(camera.cy),
    )
    with _GEOMETRY_CACHE_LOCK:
        cached = _GEOMETRY_CACHE.get(key)
        if cached is not None:
            _GEOMETRY_CACHE.move_to_end(key)
            return cached

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
            consistency = compute_depth_normal_consistency(points, scene.depth, normals)
            cons_w = 1.0 - 0.85 * consistency["disagreement"]  # type: ignore[operator]
            geometry_confidence = (np.asarray(scene.depth_confidence, dtype=np.float32) * cons_w).astype(np.float32)
    else:
        normals, geometry_confidence = estimate_normals(
            points, scene.depth, scene.depth_confidence
        )

    geometry_confidence = np.clip(geometry_confidence, 0.0, 1.0)
    geometry_confidence[~valid_depth] = 0.0
    normals_out = normals.astype(np.float32)
    conf_out = geometry_confidence.astype(np.float32)
    points.setflags(write=False)
    normals_out.setflags(write=False)
    conf_out.setflags(write=False)
    with _GEOMETRY_CACHE_LOCK:
        _GEOMETRY_CACHE[key] = (points, normals_out, conf_out)
        _GEOMETRY_CACHE.move_to_end(key)
        while len(_GEOMETRY_CACHE) > _GEOMETRY_CACHE_MAX:
            _GEOMETRY_CACHE.popitem(last=False)
    return points, normals_out, conf_out


def place_point_light_at_uv(
    scene: SceneBuffers,
    camera: CameraModel,
    u: float,
    v: float,
    *,
    standoff_fraction: float = 0.25,
    min_z: float = 0.45,
) -> tuple[float, float, float]:
    """Place a 3D point light in camera space aligned with normalized image coordinates (u, v).

    Prevents placing the bulb behind foreground subjects when (u, v) falls on distant
    background or sky pixels by clamping depth safely ahead of local/scene foreground depth.
    """
    u_c = float(np.clip(u, 0.0, 1.0))
    v_c = float(np.clip(v, 0.0, 1.0))
    h, w = scene.shape
    depth = np.asarray(scene.depth, dtype=np.float32)
    valid = (depth > 0) & np.isfinite(depth)
    if not np.any(valid):
        return (0.0, 0.0, 1.0)

    # Determine foreground-safe depth envelope excluding sky/horizon masks if present.
    from .scene_policy import detect_sky_horizon_mask
    sky_mask = detect_sky_horizon_mask(scene)
    fg_cand = valid & (scene.lightable_mask > 0.05) & (sky_mask < 0.25)
    if scene.semantic_masks:
        subj = np.zeros((h, w), dtype=bool)
        for k, m in scene.semantic_masks.items():
            kl = k.lower()
            if not any(tok in kl for tok in ("sky", "cloud", "background", "wall", "floor", "ground", "asphalt")):
                subj |= (np.asarray(m, dtype=np.float32) > 0.5)
        if np.any(subj & fg_cand):
            fg_cand = subj & fg_cand
    fg_depths = depth[fg_cand] if np.any(fg_cand) else depth[valid]
    z_fg_ref = float(np.percentile(fg_depths, 35))

    px = int(np.clip(round(u_c * (w - 1)), 0, w - 1))
    py = int(np.clip(round(v_c * (h - 1)), 0, h - 1))
    y0, y1 = max(0, py - 6), min(h, py + 7)
    x0, x1 = max(0, px - 6), min(w, px + 7)
    patch_d = depth[y0:y1, x0:x1]
    patch_valid = patch_d[patch_d > 0]
    z_local = float(np.median(patch_valid)) if patch_valid.size else z_fg_ref

    # Clamp local depth to stay in front of foreground subjects (never place bulb behind a person/face).
    z_anchor = min(z_local, z_fg_ref * 1.10)
    z_med = float(np.median(depth[valid]))
    pull = float(np.clip(float(standoff_fraction) * z_med, 0.35, 1.15))
    z_light = max(z_anchor - pull, float(min_z))

    # Project along exact camera ray for pixel (px, py) at depth z_light.
    lx = (float(px) - float(camera.cx)) / max(float(camera.fx), 1e-6) * z_light
    ly = (float(camera.cy) - float(py)) / max(float(camera.fy), 1e-6) * z_light
    return (round(lx, 4), round(ly, 4), round(z_light, 4))


def place_studio_light(
    scene: SceneBuffers,
    camera: CameraModel,
    *,
    azimuth_deg: float = -48.0,
    elevation_deg: float = 24.0,
    distance_scale: float = 1.35,
    min_z: float = 0.45,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Place an off-axis 3D studio or sky-scale light orbiting the relevant scene mass.

    For studio scenes, orbits the foreground subject so subject and background walls/floors
    react together. For macroscopic outdoor landscapes (where terrain dominates and a hiker/subject
    occupies a small foreground fraction), anchors to the full non-sky terrain volume above peak height
    so illumination covers valley and distant ridges cohesively instead of pooling inside the valley.

    Returns `(position_xyz, aim_direction_xyz)` in camera coordinates (+X right, +Y up, +Z into scene).
    """
    from .scene_policy import classify_scene_scale

    points = backproject_depth(scene.depth, camera)
    valid = (scene.depth > 0) & np.isfinite(scene.depth)
    if not np.any(valid):
        return ((-1.2, 0.8, 1.0), (0.6, -0.35, 0.71))

    scale_info = classify_scene_scale(scene)
    subj = scale_info["subject_mask"]
    env_surf = scale_info["env_surface_mask"]
    is_landscape = bool(scale_info["is_landscape"])

    if is_landscape:
        fg = valid & (scale_info["macro_anchor_mask"] > 0.25)
    else:
        fg = valid & (subj > 0.35)
        if not np.any(fg):
            fg = valid & (env_surf > 0.1)
    if not np.any(fg):
        fg = valid

    fg_pts = points[fg]
    center = np.median(fg_pts, axis=0).astype(np.float64)
    p15 = np.percentile(fg_pts, 15, axis=0)
    p85 = np.percentile(fg_pts, 85, axis=0)
    span = float(np.linalg.norm(p85 - p15))
    if is_landscape:
        # Place landscape lights at sky/sun standoff well above visible mountain ridges.
        radius = max(1.65 * span, 6.0) * float(max(distance_scale, 0.35))
        eff_el = max(float(elevation_deg), 26.0)
    else:
        radius = float(np.clip(0.55 * span, 0.75, 3.2)) * float(max(distance_scale, 0.35))
        eff_el = float(elevation_deg)

    az = np.radians(float(azimuth_deg))
    el = np.radians(eff_el)
    dir_to_light = np.array([
        np.sin(az) * np.cos(el),
        np.sin(el),
        -np.cos(az) * np.cos(el),
    ], dtype=np.float64)
    pos = center + dir_to_light * radius
    if is_landscape:
        ridge_top_y = float(np.percentile(fg_pts[:, 1], 95))
        min_sky_y = ridge_top_y + 0.35 * max(span, 1.0)
        if pos[1] < min_sky_y:
            pos[1] = min_sky_y
    if pos[2] < float(min_z):
        t = (float(min_z) - float(center[2])) / min(float(dir_to_light[2]), -1e-4)
        pos = center + dir_to_light * max(t, 0.25)
        pos[2] = float(min_z)

    aim = center - pos
    aim = aim / max(float(np.linalg.norm(aim)), 1e-8)
    pos_t = (round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4))
    aim_t = (round(float(aim[0]), 4), round(float(aim[1]), 4), round(float(aim[2]), 4))
    return pos_t, aim_t


def compute_screen_space_ao(
    points: np.ndarray,
    normals: np.ndarray,
    depth: np.ndarray,
    *,
    radii: tuple[int, ...] = (3, 9, 20),
) -> np.ndarray:
    """Compute cached multi-scale depth-horizon ambient occlusion in [0.35, 1.0] for crease/contact grounding."""
    import cv2
    from scipy.ndimage import gaussian_filter

    h, w = depth.shape
    key = (_geom_digest(depth), _geom_digest(normals), tuple(radii), h, w)
    with _GEOMETRY_CACHE_LOCK:
        hit = _AO_CACHE.get(key)
        if hit is not None:
            _AO_CACHE.move_to_end(key)
            return hit

    scale = min(1.0, 384.0 / float(max(h, w)))
    if scale < 0.99:
        hw = max(16, int(round(h * scale)))
        ww = max(16, int(round(w * scale)))
        pts_w = cv2.resize(points, (ww, hw), interpolation=cv2.INTER_AREA)
        nrm_w = cv2.resize(normals, (ww, hw), interpolation=cv2.INTER_AREA)
        nrm_w = nrm_w / np.maximum(np.linalg.norm(nrm_w, axis=-1, keepdims=True), 1e-8)
        dep_w = cv2.resize(depth, (ww, hw), interpolation=cv2.INTER_AREA)
        eff_radii = tuple(max(1, int(round(r * scale))) for r in radii)
    else:
        hw, ww = h, w
        pts_w, nrm_w, dep_w = points, normals, depth
        eff_radii = radii

    valid = dep_w > 0
    if not np.any(valid):
        out = np.ones((h, w), dtype=np.float32)
        out.setflags(write=False)
        return out

    z_med = max(float(np.median(dep_w[valid])), 1e-3)
    max_r = max(0.35 * z_med, 0.40)
    occ_sum = np.zeros((hw, ww), dtype=np.float32)
    count = 0

    for rad in eff_radii:
        diag = max(1, int(round(rad * 0.7071)))
        offsets = (
            (0, rad), (0, -rad), (rad, 0), (-rad, 0),
            (diag, diag), (diag, -diag), (-diag, diag), (-diag, -diag),
        )
        for dy, dx in offsets:
            pts_n = np.roll(np.roll(pts_w, dy, axis=0), dx, axis=1)
            val_n = np.roll(np.roll(valid, dy, axis=0), dx, axis=1).copy()
            if dy > 0:
                val_n[:dy, :] = False
            elif dy < 0:
                val_n[dy:, :] = False
            if dx > 0:
                val_n[:, :dx] = False
            elif dx < 0:
                val_n[:, dx:] = False
            delta = pts_n - pts_w
            dist = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 1e-8))
            sin_elev = np.sum(nrm_w * (delta / dist[..., None]), axis=-1)
            range_w = np.clip(1.0 - dist / max_r, 0.0, 1.0)
            hit = np.where(valid & val_n & (sin_elev > 0.12), np.clip((sin_elev - 0.12) * 1.15, 0.0, 1.0) * range_w, 0.0)
            occ_sum += hit.astype(np.float32)
            count += 1

    ao_raw = 1.0 - np.clip(occ_sum / max(float(count) * 0.25, 1.0), 0.0, 0.65)
    ao_lo = np.clip(gaussian_filter(ao_raw, 1.2, mode="reflect"), 0.35, 1.0).astype(np.float32)
    out = cv2.resize(ao_lo, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) if scale < 0.99 else ao_lo
    out.setflags(write=False)
    with _GEOMETRY_CACHE_LOCK:
        _AO_CACHE[key] = out
        _AO_CACHE.move_to_end(key)
        while len(_AO_CACHE) > _GEOMETRY_CACHE_MAX:
            _AO_CACHE.popitem(last=False)
    return out
