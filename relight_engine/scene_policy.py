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
