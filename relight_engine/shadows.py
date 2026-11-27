from __future__ import annotations
import numpy as np
import cv2
from .contracts import CameraModel, SceneBuffers, EngineSettings


def _foreground_silhouette_rim(depth: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Detect curved foreground silhouette edges overlooking deeper background so limbs do not self-outline."""
    valid = depth > 0
    max_bg = np.zeros_like(depth, dtype=np.float32)
    for dy, dx in (
        (-2, 0), (2, 0), (0, -2), (0, 2),
        (-5, 0), (5, 0), (0, -5), (0, 5),
        (-9, 0), (9, 0), (0, -9), (0, 9),
        (-13, 0), (13, 0), (0, -13), (0, 13),
    ):
        dn = np.roll(np.roll(depth, dy, axis=0), dx, axis=1)
        vn = np.roll(np.roll(valid, dy, axis=0), dx, axis=1).copy()
        if dy > 0:
            vn[:dy, :] = False
        elif dy < 0:
            vn[dy:, :] = False
        if dx > 0:
            vn[:, :dx] = False
        elif dx < 0:
            vn[:, dx:] = False
        cand = np.where(valid & vn, dn, 0.0)
        max_bg = np.maximum(max_bg, cand)
    fg_drop_rel = np.where(valid, np.maximum(max_bg - depth, 0.0) / np.maximum(depth, 1e-3), 0.0)
    # Curved/grazing limbs overlooking a deeper jump qualify; flat background walls/floors stay at 0.
    rim_jump = np.clip((fg_drop_rel - 0.03) / 0.11, 0.0, 1.0)
    curved = np.clip((0.975 - np.abs(normals[..., 2])) / 0.35, 0.0, 1.0)
    return (rim_jump * (0.40 + 0.60 * curved)).astype(np.float32)


def _continuous_surface_slope(depth: np.ndarray) -> np.ndarray:
    """Estimate per-pixel continuous surface depth slope without bleeding across silhouette jumps.

    Uses the minimum magnitude of one-sided finite differences along each axis so flat
    receivers immediately adjacent to an occluder keep zero slope inflation (preserving
    tight contact shadows) while curved/foreshortened surfaces scale smoothly.
    """
    valid = depth > 0
    grad_sq = np.zeros_like(depth, dtype=np.float32)
    for axis in (0, 1):
        d_m = np.roll(depth, 1, axis=axis)
        d_p = np.roll(depth, -1, axis=axis)
        v_m = np.roll(valid, 1, axis=axis).copy()
        v_p = np.roll(valid, -1, axis=axis).copy()
        if axis == 0:
            v_m[0, :] = False
            v_p[-1, :] = False
        else:
            v_m[:, 0] = False
            v_p[:, -1] = False
        g_m = np.where(valid & v_m, np.abs(depth - d_m), np.inf)
        g_p = np.where(valid & v_p, np.abs(d_p - depth), np.inf)
        g_min = np.minimum(g_m, g_p)
        g_min = np.where(np.isfinite(g_min), g_min, 0.0)
        g_min = np.minimum(g_min, 0.04 * np.maximum(depth, 1e-3))
        grad_sq += g_min * g_min
    return np.sqrt(grad_sq).astype(np.float32)


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


def _pcss_penumbra_smooth(visibility: np.ndarray, depth: np.ndarray,
                          penumbra_px: np.ndarray) -> np.ndarray:
    """Blend depth-aware multi-scale visibility for PCSS-style contact-sharp to far-soft shadows."""
    from scipy.ndimage import gaussian_filter

    valid = depth > 0
    log_d = np.log(np.maximum(depth, 1e-3)).astype(np.float32)

    def _guided_vis(sig: float) -> np.ndarray:
        mean_i = gaussian_filter(log_d, sig, mode='reflect')
        mean_ii = gaussian_filter(log_d * log_d, sig, mode='reflect')
        var_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)
        mean_p = gaussian_filter(visibility, sig, mode='reflect')
        mean_ip = gaussian_filter(log_d * visibility, sig, mode='reflect')
        cov_ip = mean_ip - mean_i * mean_p
        a = cov_ip / (var_i + 0.005)
        b = mean_p - a * mean_i
        out = gaussian_filter(a, sig, mode='reflect') * log_d + gaussian_filter(b, sig, mode='reflect')
        return np.clip(np.where(valid, out, visibility), 0.0, 1.0).astype(np.float32)

    v_med = _guided_vis(2.0)
    v_wide = _guided_vis(5.0)
    t1 = np.clip(penumbra_px / 2.5, 0.0, 1.0)
    t2 = np.clip((penumbra_px - 2.5) / 4.5, 0.0, 1.0)
    blended = visibility * (1.0 - t1) + v_med * t1
    return np.clip(blended * (1.0 - t2) + v_wide * t2, 0.0, 1.0).astype(np.float32)


def _estimate_blocker_world_width(depth: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Estimate local foreground silhouette thickness in world depth units around depth jumps."""
    h, w = depth.shape
    k = max(11, (min(h, w) // 12) | 1)
    k = min(k, 45)
    local_bg = cv2.dilate(depth, np.ones((k, k), dtype=np.uint8))
    fg_step = (depth > 0) & ((local_bg - depth) > (0.22 * np.maximum(depth, 1e-3)))
    if not np.any(fg_step):
        return np.full((h, w), max(float(np.median(depth[depth > 0])) if np.any(depth > 0) else 1.0, 0.5), dtype=np.float32)
    dist_px = cv2.distanceTransform(fg_step.astype(np.uint8), cv2.DIST_L2, 3)
    # Dilate slightly so silhouette rim pixels inherit interior thickness cleanly
    rim_px = cv2.dilate(dist_px, np.ones((5, 5), dtype=np.uint8))
    world_w = ((2.0 * rim_px + 2.0) * depth / max(float(camera.fx), 1.0)).astype(np.float32)
    return np.where(fg_step, np.maximum(world_w, 0.04), np.maximum(depth * 0.5, 0.5)).astype(np.float32)


def ray_visibility(points: np.ndarray, normals: np.ndarray, depth: np.ndarray,
                   camera: CameraModel, to_light: np.ndarray,
                   max_distance: np.ndarray, depth_kind: str,
                   settings: EngineSettings,
                   light_radius: float = 0.0,
                   depth_confidence: np.ndarray | None = None,
                   organic_receiver_mask: np.ndarray | None = None) -> np.ndarray:
    """Approximate 2.5D visibility by marching camera-projected rays through the depth map.

    Features grazing- and slope-adaptive self-shadow bias, confidence-aware tolerances,
    vectorized 2D ray stepping, optional coarse-to-fine adaptive step allocation,
    multi-pass edge-aware Bayer smoothing, and PCSS-style penumbra softening.
    """
    h, w = depth.shape
    norm_len = np.sqrt(np.sum(to_light * to_light, axis=-1, keepdims=True))
    direction = to_light / np.maximum(norm_len, 1e-8)
    base_bias = settings.shadow_bias_metric if depth_kind == 'metric' else settings.shadow_bias_relative
    depth_f32 = depth.astype(np.float32, copy=False)
    if depth_kind == 'relative':
        valid_d = depth_f32 > 0
        if np.any(valid_d):
            med_d = float(np.median(depth_f32[valid_d]))
            if med_d < 1.0:
                z_scale = float(np.clip(med_d, 0.4, 1.0))
            elif med_d > 5.0:
                z_scale = float(np.clip(med_d / 5.0, 1.0, 3.0))
            else:
                z_scale = 1.0
            base_bias = base_bias * z_scale

    ndotl = np.clip(np.sum(normals * direction, axis=-1), 0.0, 1.0)
    grazing = 1.0 - ndotl
    cont_dz = _continuous_surface_slope(depth_f32)
    fg_rim_w = _foreground_silhouette_rim(depth_f32, normals)
    if depth_confidence is not None:
        conf_factor = 1.0 + 0.35 * (1.0 - np.clip(depth_confidence.astype(np.float32, copy=False), 0.0, 1.0))
    else:
        conf_factor = np.float32(1.0)

    bias_map = (
        base_bias * (1.0 + 3.0 * (grazing * grazing) + 2.5 * fg_rim_w) * conf_factor
        + 1.35 * cont_dz * (0.4 + 0.6 * grazing)
    ).astype(np.float32)
    normal_offset = (base_bias * (2.0 + 1.5 * (grazing * grazing) + 2.5 * fg_rim_w)).astype(np.float32)

    ox = (points[..., 0] + normals[..., 0] * normal_offset).astype(np.float32)
    oy = (points[..., 1] + normals[..., 1] * normal_offset).astype(np.float32)
    oz = (points[..., 2] + normals[..., 2] * normal_offset).astype(np.float32)
    dx = (direction[..., 0] * max_distance).astype(np.float32)
    dy = (direction[..., 1] * max_distance).astype(np.float32)
    dz = (direction[..., 2] * max_distance).astype(np.float32)

    visibility = np.ones((h, w), np.float32)
    blocker_step = np.full((h, w), 1.0, np.float32)
    max_penetration = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    bayer = (((yy & 1) * 2 + (xx & 1)) * 0.25).astype(np.float32)

    n_steps = max(2, settings.shadow_steps)
    # Near-field start at 0.025 (with u^1.8 march below) captures tight contact shadows cleanly across blockers.
    step_base = np.linspace(0.025, 0.96, n_steps, dtype=np.float32)
    delta_u = (0.96 - 0.025) / max(n_steps - 1, 1)
    jitter = ((bayer - 0.375) * (delta_u * 0.85)).astype(np.float32)
    inv_soft_width = (1.0 / np.maximum(bias_map * 3.0, 1e-4)).astype(np.float32)

    fx = np.float32(camera.fx)
    fy = np.float32(camera.fy)
    cx = np.float32(camera.cx)
    cy = np.float32(camera.cy)
    max_t = float(settings.shadow_max_thickness) if settings.shadow_max_thickness is not None else None
    inv_fade_w = np.float32(1.0 / max(max_t * 0.35, 1e-4)) if max_t is not None else None
    blocker_w_map = (
        _estimate_blocker_world_width(depth_f32, camera)
        if (max_t is not None and getattr(settings, "shadow_width_aware", False))
        else None
    )

    def _march_steps(base_u_list: np.ndarray, mask: np.ndarray | None = None) -> None:
        nonlocal visibility, blocker_step, max_penetration
        if mask is None:
            for base_u in base_u_list:
                u_step = np.clip(np.float32(base_u) + jitter, 0.02, 0.98)
                step = u_step * np.power(u_step, 0.8, dtype=np.float32)
                qz = oz + dz * step
                safe_z = np.maximum(qz, 1e-5)
                inv_z = np.float32(1.0) / safe_z
                map_x = fx * (ox + dx * step) * inv_z + cx
                map_y = cy - fy * (oy + dy * step) * inv_z
                sampled = cv2.remap(depth_f32, map_x, map_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                inside = (
                    (map_x >= 1.0) & (map_x < (w - 1))
                    & (map_y >= 1.0) & (map_y < (h - 1))
                    & (qz > 0.0) & (sampled > 0.0)
                )
                penetration = (qz - bias_map) - sampled
                valid_hit = inside & (penetration > 0.0)
                if not np.any(valid_hit):
                    continue
                raw_hit = np.where(valid_hit, np.minimum(penetration * inv_soft_width, 1.0), 0.0)
                if max_t is not None:
                    fade = np.clip((max_t - penetration) * inv_fade_w, 0.0, 1.0)
                    cand = valid_hit & (penetration < max_t)
                    if np.any(cand):
                        sy = np.clip(np.rint(map_y[cand]).astype(np.intp), 0, h - 1)
                        sx = np.clip(np.rint(map_x[cand]).astype(np.intp), 0, w - 1)
                        near_d = depth_f32[sy, sx]
                        near_pen = (qz[cand] - bias_map[cand]) - near_d
                        cliff_diff = np.abs(sampled[cand] - near_d)
                        phantom = (
                            ((cliff_diff > np.maximum(0.06 * near_d, 0.05)) & (near_pen <= 0.0))
                            | ((cliff_diff > max_t * 0.5) & (near_pen >= max_t))
                            | ((step[cand] < 0.015) & (cliff_diff > np.maximum(0.04 * near_d, 0.03)))
                        )
                        f_c = fade[cand]
                        if blocker_w_map is not None:
                            local_t = np.minimum(max_t, 1.15 * blocker_w_map[sy, sx] + 0.04 * near_d)
                            f_w = np.clip((local_t - penetration[cand]) / np.maximum(local_t * 0.35, 1e-4), 0.0, 1.0)
                            f_c = np.minimum(f_c, f_w)
                        if np.any(phantom):
                            f_c[phantom] = 0.0
                        fade[cand] = f_c
                    raw_hit *= fade
                hit_mask = raw_hit > 0.15
                blocker_step = np.where(hit_mask, np.minimum(blocker_step, step), blocker_step)
                max_penetration = np.where(hit_mask, np.maximum(max_penetration, penetration), max_penetration)
                visibility = np.minimum(visibility, 1.0 - raw_hit)
                if not np.any(visibility > 0.02):
                    break
        else:
            ys_m, xs_m = np.where(mask)
            n_pts = len(ys_m)
            if n_pts == 0:
                return
            ox_m, oy_m, oz_m = ox[ys_m, xs_m], oy[ys_m, xs_m], oz[ys_m, xs_m]
            dx_m, dy_m, dz_m = dx[ys_m, xs_m], dy[ys_m, xs_m], dz[ys_m, xs_m]
            jit_m = jitter[ys_m, xs_m]
            bias_m = bias_map[ys_m, xs_m]
            inv_sw_m = inv_soft_width[ys_m, xs_m]
            vis_m = visibility[ys_m, xs_m]
            blk_m = blocker_step[ys_m, xs_m]
            pen_m = max_penetration[ys_m, xs_m]

            # Reshape masked rays into (rows_blk, w) blocks so cv2.remap stays within SHRT_MAX bounds.
            pad_n = ((n_pts + w - 1) // w) * w
            mx_buf = np.zeros(pad_n, dtype=np.float32)
            my_buf = np.zeros(pad_n, dtype=np.float32)
            rows_blk = pad_n // w
            for base_u in base_u_list:
                u_step = np.clip(np.float32(base_u) + jit_m, 0.02, 0.98)
                step = u_step * np.power(u_step, 0.8, dtype=np.float32)
                qz = oz_m + dz_m * step
                safe_z = np.maximum(qz, 1e-5)
                inv_z = np.float32(1.0) / safe_z
                mx = fx * (ox_m + dx_m * step) * inv_z + cx
                my = cy - fy * (oy_m + dy_m * step) * inv_z
                mx_buf[:n_pts] = mx
                my_buf[:n_pts] = my
                sampled = cv2.remap(
                    depth_f32, mx_buf.reshape(rows_blk, w), my_buf.reshape(rows_blk, w),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                ).ravel()[:n_pts]
                inside = (
                    (mx >= 1.0) & (mx < (w - 1))
                    & (my >= 1.0) & (my < (h - 1))
                    & (qz > 0.0) & (sampled > 0.0)
                )
                pen = (qz - bias_m) - sampled
                vh = inside & (pen > 0.0)
                if not np.any(vh):
                    continue
                rh = np.where(vh, np.minimum(pen * inv_sw_m, 1.0), 0.0)
                if max_t is not None:
                    fade_m = np.clip((max_t - pen) * inv_fade_w, 0.0, 1.0)
                    cand_m = vh & (pen < max_t)
                    if np.any(cand_m):
                        sy_m = np.clip(np.rint(my[cand_m]).astype(np.intp), 0, h - 1)
                        sx_m = np.clip(np.rint(mx[cand_m]).astype(np.intp), 0, w - 1)
                        near_dm = depth_f32[sy_m, sx_m]
                        near_pm = (qz[cand_m] - bias_m[cand_m]) - near_dm
                        cliff_dm = np.abs(sampled[cand_m] - near_dm)
                        phantom_m = (
                            ((cliff_dm > np.maximum(0.06 * near_dm, 0.05)) & (near_pm <= 0.0))
                            | ((cliff_dm > max_t * 0.5) & (near_pm >= max_t))
                            | ((step[cand_m] < 0.015) & (cliff_dm > np.maximum(0.04 * near_dm, 0.03)))
                        )
                        f_cm = fade_m[cand_m]
                        if blocker_w_map is not None:
                            local_tm = np.minimum(max_t, 1.15 * blocker_w_map[sy_m, sx_m] + 0.04 * near_dm)
                            f_wm = np.clip((local_tm - pen[cand_m]) / np.maximum(local_tm * 0.35, 1e-4), 0.0, 1.0)
                            f_cm = np.minimum(f_cm, f_wm)
                        if np.any(phantom_m):
                            f_cm[phantom_m] = 0.0
                        fade_m[cand_m] = f_cm
                    rh *= fade_m
                hm = rh > 0.15
                blk_m = np.where(hm, np.minimum(blk_m, step), blk_m)
                pen_m = np.where(hm, np.maximum(pen_m, pen), pen_m)
                vis_m = np.minimum(vis_m, 1.0 - rh)
            visibility[ys_m, xs_m] = vis_m
            blocker_step[ys_m, xs_m] = blk_m
            max_penetration[ys_m, xs_m] = pen_m

    if bool(settings.shadow_adaptive_sampling) and n_steps >= 16:
        coarse_steps = step_base[::2]
        fine_steps = step_base[1::2]
        _march_steps(coarse_steps)
        in_penumbra = (visibility > 0.02) & (visibility < 0.98)
        shadow_edge = cv2.dilate((visibility < 0.98).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        near_shadow_lit = shadow_edge & (visibility > 0.02)
        _march_steps(fine_steps, mask=(in_penumbra | near_shadow_lit))
    else:
        _march_steps(step_base)

    vis_out = visibility
    for _ in range(max(1, int(settings.shadow_smooth_passes))):
        vis_out = _smooth_visibility_edge_aware(vis_out, depth_f32)
    if float(light_radius) > 0.0 and np.any(vis_out < 0.99):
        # Blocker-to-receiver gap controls PCSS penumbra width, including physical depth separation
        # across foreground-to-background cliffs so background borders don't catch tight dark ink halos.
        gap_ratio = blocker_step / np.maximum(1.0 - blocker_step, 0.15)
        depth_gap_ratio = max_penetration / np.maximum(depth_f32 * 0.45, 0.25)
        eff_gap = np.maximum(gap_ratio, depth_gap_ratio)
        penumbra_px = np.clip(float(light_radius) * eff_gap * 22.0, 0.0, 14.0)
        vis_out = _pcss_penumbra_smooth(vis_out, depth_f32, penumbra_px)
    if depth_confidence is not None:
        # Soften false deep shadow halos on uncertain occlusion/silhouette depth pixels.
        occ_floor = np.where(depth_confidence < 0.25, np.clip(1.0 - depth_confidence * 3.0, 0.25, 0.75), 0.0).astype(np.float32)
        vis_out = np.maximum(vis_out, occ_floor)
    if np.any(fg_rim_w > 0.05):
        # Prevent curved foreground silhouette lips (e.g. fingers overlooking a distant backdrop
        # or curved limbs next to raised watches) from catching harsh dark shadow markings.
        sil_floor = np.clip(fg_rim_w * (0.68 + 0.22 * grazing), 0.0, 0.88).astype(np.float32)
        vis_out = np.maximum(vis_out, sil_floor)
    if float(light_radius) >= 0.12:
        # Under soft extended studio lights, near-field shallow self-occlusion on curved organic surfaces
        # or foreground-on-foreground relief (e.g. wrist/strap/knuckles onto fingers, vs cast shadows onto
        # distant walls/floors) wraps smoothly instead of stamping dark blotches across fingertips/crevices.
        curved_fg = np.clip((0.92 - np.abs(normals[..., 2])) / 0.35, 0.0, 1.0)
        near_self = np.clip((0.18 - blocker_step) / 0.15, 0.0, 1.0)
        wrap_floor = curved_fg * near_self * min(float(light_radius) * 2.2, 0.65)
        valid_d_vis = depth_f32 > 0
        if np.any(valid_d_vis):
            d_lo_ref, d_hi_ref = np.quantile(depth_f32[valid_d_vis], [0.15, 0.88])
            span_d = float(d_hi_ref - d_lo_ref)
            if span_d > 0.35 * max(float(d_lo_ref), 0.5):
                # Connected foreground manifold elevation: 0 on distant walls/floors, -> 1 on foreground hands/objects.
                fg_elev = np.clip((float(d_hi_ref) - depth_f32) / max(span_d, 0.25), 0.0, 1.0)
                # Broad foreground self-occlusion relief (watch->finger, finger->finger) vs long room wall cast.
                close_occl = np.clip((0.58 - blocker_step) / 0.36, 0.0, 1.0)
                fg_self_floor = np.clip(fg_elev * close_occl * min(float(light_radius) * 3.5, 0.88), 0.0, 0.88)
                wrap_floor = np.maximum(wrap_floor, fg_self_floor)
        if organic_receiver_mask is not None:
            org_w = np.clip(np.asarray(organic_receiver_mask, dtype=np.float32), 0.0, 1.0)
            org_floor = np.clip(org_w * min(float(light_radius) * 3.2, 0.86), 0.0, 0.86)
            wrap_floor = np.maximum(wrap_floor, org_floor)
        vis_out = np.maximum(vis_out, wrap_floor.astype(np.float32))
    return vis_out


def scene_shadow_visibility(scene: SceneBuffers, camera: CameraModel, points: np.ndarray,
                            normals: np.ndarray, light_positions: np.ndarray | None,
                            directional_vectors: np.ndarray | None,
                            settings: EngineSettings,
                            light_radius: float = 0.0) -> np.ndarray:
    h, w = scene.depth.shape
    org_mask = np.zeros((h, w), dtype=np.float32)
    if scene.subsurface is not None:
        org_mask = np.maximum(org_mask, np.clip(np.asarray(scene.subsurface, dtype=np.float32) * 2.0, 0.0, 1.0))
    if scene.semantic_masks:
        for mk, mv in scene.semantic_masks.items():
            if any(t in mk.lower() for t in ("hand", "finger", "skin", "palm", "wrist")):
                org_mask = np.maximum(org_mask, np.clip(np.asarray(mv, dtype=np.float32), 0.0, 1.0))
    org_arg = org_mask if np.any(org_mask > 1e-3) else None

    if light_positions is not None:
        vec = light_positions - points
        dist = np.linalg.norm(vec, axis=-1)
        vis = ray_visibility(
            points, normals, scene.depth, camera, vec, dist,
            scene.depth_kind, settings, light_radius=light_radius,
            depth_confidence=scene.depth_confidence,
            organic_receiver_mask=org_arg,
        )
    else:
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
        vis = ray_visibility(
            points, normals, scene.depth, camera, directional_vectors, max_dist,
            scene.depth_kind, settings, light_radius=light_radius,
            depth_confidence=scene.depth_confidence,
            organic_receiver_mask=org_arg,
        )

    if getattr(settings, "protect_sky", True):
        from .scene_policy import detect_sky_horizon_mask
        sky_w = detect_sky_horizon_mask(scene)
        if np.any(sky_w > 1e-4):
            vis = (vis * (1.0 - sky_w) + sky_w).astype(np.float32)
    return vis
