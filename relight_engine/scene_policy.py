"""Image-domain agnostic semantic-mask policy for Relight Studio.

Semantic labels are data, never a closed taxonomy. The same policy works for
"person", "glass bottle", "sofa", "cat", "building", or labels not known
when the engine was written.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence
import numpy as np


def semantic_key(label: str) -> str:
    """Stable mask key for a free-form noun phrase or object label."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Semantic labels must be non-empty strings.")
    key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    if not key:
        raise ValueError(f"Could not normalize semantic label: {label!r}")
    return key


@dataclass(frozen=True)
class SemanticRoutingPolicy:
    """User/product policy for semantic masks.

    None for lightable_labels means all visible pixels are eligible by default;
    geometry confidence and explicit protected masks still gate the renderer.
    An explicit tuple restricts relighting to those semantic labels.
    No class (for example sky, water, skin or glass) is hardcoded as protected.
    """
    protected_labels: tuple[str, ...] = ()
    lightable_labels: tuple[str, ...] | None = None

    def validate(self) -> None:
        for label in self.protected_labels:
            semantic_key(label)
        if self.lightable_labels is not None:
            if not self.lightable_labels:
                raise ValueError("lightable_labels cannot be empty; use None for all visible pixels.")
            for label in self.lightable_labels:
                semantic_key(label)

    def to_metadata(self) -> dict:
        return {
            "protected_labels": [semantic_key(x) for x in self.protected_labels],
            "lightable_labels": None if self.lightable_labels is None else [semantic_key(x) for x in self.lightable_labels],
            "default_policy": "all_visible_pixels_except_explicit_protection",
        }


def compose_scene_masks(
    masks: Mapping[str, np.ndarray],
    *,
    shape: tuple[int, int] | None = None,
    policy: SemanticRoutingPolicy | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose generic lightable/protected masks from arbitrary semantic labels.

    Masks are soft confidence/coverage maps in [0, 1]. A class may be any string.
    The default behavior is intentionally neutral: all pixels are candidates until a
    user or future scene-understanding policy explicitly protects/restricts a region.
    Geometry/material confidence is applied later by the rendering engine.
    """
    policy = policy or SemanticRoutingPolicy()
    policy.validate()
    normalized: dict[str, np.ndarray] = {}
    inferred_shape = shape
    for label, raw in masks.items():
        key = semantic_key(label)
        if key in normalized:
            raise ValueError(f"Duplicate semantic mask key after normalization: {key}")
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Semantic mask {label!r} must be a 2D HxW array.")
        if inferred_shape is None:
            inferred_shape = array.shape
        if array.shape != inferred_shape:
            raise ValueError("All semantic masks must share the same HxW image grid.")
        if not np.isfinite(array).all():
            raise ValueError(f"Semantic mask {label!r} contains non-finite values.")
        normalized[key] = np.clip(array, 0.0, 1.0)
    if inferred_shape is None:
        raise ValueError("shape is required when no semantic masks are available.")

    zero = np.zeros(inferred_shape, dtype=np.float32)
    protected = zero.copy()
    for label in policy.protected_labels:
        protected = np.maximum(protected, normalized.get(semantic_key(label), zero))

    if policy.lightable_labels is None:
        base = np.ones(inferred_shape, dtype=np.float32)
    else:
        base = zero.copy()
        for label in policy.lightable_labels:
            base = np.maximum(base, normalized.get(semantic_key(label), zero))

    lightable = base * (1.0 - protected)
    return np.clip(lightable, 0.0, 1.0), np.clip(protected, 0.0, 1.0)


def detect_sky_horizon_mask(scene) -> np.ndarray:
    """Detect infinite-depth sky / cloud horizon pixels that must not catch foreground point wash or cast shadows.

    Combines explicit sky/cloud semantic masks (when present) with a conservative geometric
    far-plane upper-frame saturation check on relative-depth scenes.
    """
    from scipy.ndimage import gaussian_filter

    h, w = scene.shape
    sky = np.zeros((h, w), dtype=np.float32)
    for k, mask in scene.semantic_masks.items():
        kl = k.lower()
        if any(tok in kl for tok in ("sky", "cloud", "heaven")):
            sky = np.maximum(sky, np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0))

    depth = np.asarray(scene.depth, dtype=np.float32)
    valid = depth > 0
    if scene.depth_kind == "relative" and np.any(valid):
        d_vals = depth[valid]
        d_min = float(np.min(d_vals))
        d_max = float(np.max(d_vals))
        if d_max - d_min > 0.5 and d_max >= 4.0:
            gy = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
            gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
            d_grad = np.hypot(gx, gy)
            upper = (np.arange(h, dtype=np.float32)[:, None] / max(float(h - 1), 1.0)) < 0.62
            sat_far = valid & (depth >= d_max * 0.965) & (d_grad < 0.04 * d_max)
            # Only treat unlabeled saturated far depth as geometric sky when it touches top edge,
            # does not extend across bottom ground rows, and is RGB-smooth (not a wall/checkerboard).
            top_far_frac = float(np.mean(sat_far[0, :])) if h > 0 else 0.0
            bot_far_frac = float(np.mean(sat_far[-1, :])) if h > 0 else 0.0
            if top_far_frac > 0.15 and bot_far_frac < 0.15:
                rgb_luma = 0.2126 * scene.rgb_srgb[..., 0] + 0.7152 * scene.rgb_srgb[..., 1] + 0.0722 * scene.rgb_srgb[..., 2]
                rgb_hf = np.abs(rgb_luma - gaussian_filter(rgb_luma, 1.5, mode="reflect"))
                cand = sat_far & upper & (rgb_hf < 0.03)
                if np.any(cand):
                    sat_smooth = gaussian_filter(cand.astype(np.float32), 2.5, mode="reflect")
                    sky = np.maximum(sky, np.clip(sat_smooth, 0.0, 1.0))

    if np.any(sky > 1e-4):
        sky = np.clip(gaussian_filter(sky, 1.5, mode="reflect"), 0.0, 1.0).astype(np.float32)
    return sky


def classify_scene_layers(scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify finite scene into (subject_mask, env_surface_mask, sky_mask) in [0, 1].

    Distinguishes foreground subjects from physical 3D walls/floors/grounds (`env_surface_mask`),
    both of which participate in 3D room lighting and bounce, versus infinite `sky_mask`.
    """
    from scipy.ndimage import gaussian_filter

    h, w = scene.shape
    sky = detect_sky_horizon_mask(scene)
    subj = np.zeros((h, w), dtype=np.float32)

    env_tokens = (
        "sky", "cloud", "heaven", "background", "wall", "floor", "ground",
        "asphalt", "road", "pavement", "grass", "mountain", "water", "sea",
    )
    if scene.semantic_masks:
        for k, m in scene.semantic_masks.items():
            kl = k.lower()
            if not any(tok in kl for tok in env_tokens):
                subj = np.maximum(subj, np.clip(np.asarray(m, dtype=np.float32), 0.0, 1.0))

    depth = np.asarray(scene.depth, dtype=np.float32)
    valid = (depth > 0) & (sky < 0.5)
    if not np.any(subj > 0.15) and np.any(valid):
        z_vals = depth[valid]
        q15, q65 = np.quantile(z_vals, [0.15, 0.65])
        if float(q65 - q15) > 0.25:
            subj = np.clip((float(q65) - depth) / max(float(q65 - q15), 1e-3), 0.0, 1.0)

    subj = np.clip(gaussian_filter(subj, 2.0, mode="reflect") * (1.0 - sky), 0.0, 1.0).astype(np.float32)
    env_surface = np.clip((1.0 - sky) * (1.0 - subj), 0.0, 1.0).astype(np.float32)
    return subj, env_surface, sky


def classify_scene_scale(scene) -> dict[str, object]:
    """Distinguish macroscopic outdoor landscapes from close studio scenes for scale-aware lighting.

    A scene is treated as `landscape` when terrain/environment covers most of the frame while
    foreground segmented subjects occupy only a small minority (`< 10%`), accompanied by either
    landscape semantic masks or significant sky + depth range.
    """
    subj, env_surf, sky = classify_scene_layers(scene)
    sky_frac = float(np.mean(sky > 0.5))
    subj_frac = float(np.mean(subj > 0.35))
    env_frac = float(np.mean(env_surf > 0.35))

    landscape_tokens = (
        "mountain", "majestic_mountains", "valley", "glacier", "hill",
        "cliff", "canyon", "ocean", "sea", "landscape", "terrain",
    )
    has_landscape_sem = False
    if getattr(scene, "semantic_masks", None):
        for k, m in scene.semantic_masks.items():
            kl = k.lower()
            if any(tok in kl for tok in landscape_tokens) and np.any(np.asarray(m) > 0.10):
                has_landscape_sem = True
                break

    depth = np.asarray(scene.depth, dtype=np.float32)
    valid = (depth > 0) & np.isfinite(depth)
    if np.any(valid):
        z_vals = depth[valid]
        z_p05, z_p95 = np.quantile(z_vals, [0.05, 0.95])
        depth_span_ratio = float(z_p95 / max(float(z_p05), 1e-3))
    else:
        depth_span_ratio = 1.0

    is_landscape = bool(
        env_frac >= 0.40
        and subj_frac < 0.10
        and (has_landscape_sem or (sky_frac >= 0.12 and depth_span_ratio >= 2.2))
    )
    macro_anchor_mask = np.clip((env_surf + subj) * (1.0 - sky), 0.0, 1.0).astype(np.float32)
    return {
        "scene_scale": "landscape" if is_landscape else "studio",
        "is_landscape": is_landscape,
        "sky_frac": sky_frac,
        "subj_frac": subj_frac,
        "env_frac": env_frac,
        "depth_span_ratio": depth_span_ratio,
        "subject_mask": subj,
        "env_surface_mask": env_surf,
        "sky_mask": sky,
        "macro_anchor_mask": macro_anchor_mask,
    }
