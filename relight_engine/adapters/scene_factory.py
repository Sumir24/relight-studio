from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ..contracts import SceneBuffers
from .depth_anything_hf import DepthPrediction

@dataclass
class SemanticPrediction:
    lightable_mask: np.ndarray | None = None
    roughness: np.ndarray | None = None
    metalness: np.ndarray | None = None
    specular_level: np.ndarray | None = None
    protected_mask: np.ndarray | None = None
    class_masks: dict[str, np.ndarray] = field(default_factory=dict)
    instance_masks: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def build_scene_buffers(rgb_srgb: np.ndarray, depth: DepthPrediction,
                        semantics: SemanticPrediction, alpha_mask: np.ndarray | None = None) -> SceneBuffers:
    """Join independently estimated depth and semantic/material outputs on one pixel grid.

    Inputs must already be registered to the exact RGB resolution. No silent resize is done.
    """
    if rgb_srgb.ndim != 3 or rgb_srgb.shape[-1] != 3:
        raise ValueError("rgb_srgb must have shape HxWx3.")
    h, w = rgb_srgb.shape[:2]
    if depth.depth.shape != (h, w) or depth.confidence.shape != (h, w):
        raise ValueError('Depth outputs must exactly match RGB resolution; resize with a documented transform first.')
    if semantics.lightable_mask is not None and semantics.lightable_mask.shape != (h, w):
        raise ValueError('Semantic masks must exactly match RGB resolution.')
    if semantics.protected_mask is not None and semantics.protected_mask.shape != (h, w):
        raise ValueError('Protected mask must exactly match RGB resolution.')
    if alpha_mask is not None and np.asarray(alpha_mask).shape != (h, w):
        raise ValueError("alpha_mask must exactly match RGB resolution.")
    for family_name, masks in (("class", semantics.class_masks), ("instance", semantics.instance_masks)):
        for name, raw_mask in masks.items():
            mask = np.asarray(raw_mask)
            if mask.shape != (h, w):
                raise ValueError(f'{family_name.capitalize()} mask {name!r} must exactly match RGB resolution.')
            if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
                raise ValueError(f'{family_name.capitalize()} mask {name!r} must be finite and in [0,1].')
    return SceneBuffers(
        rgb_srgb=rgb_srgb.astype(np.float32),
        depth=depth.depth.astype(np.float32),
        depth_confidence=depth.confidence.astype(np.float32),
        lightable_mask=(np.ones((h, w), dtype=np.float32) if semantics.lightable_mask is None
                        else semantics.lightable_mask.astype(np.float32)),
        depth_kind=depth.kind,
        alpha_mask=None if alpha_mask is None else np.asarray(alpha_mask, dtype=np.float32),
        protected_mask=None if semantics.protected_mask is None else semantics.protected_mask.astype(np.float32),
        semantic_masks={k: np.asarray(v, dtype=np.float32) for k, v in semantics.class_masks.items()},
        instance_masks={k: np.asarray(v, dtype=np.float32) for k, v in semantics.instance_masks.items()},
        roughness=None if semantics.roughness is None else semantics.roughness.astype(np.float32),
        metalness=None if semantics.metalness is None else semantics.metalness.astype(np.float32),
        specular_level=None if semantics.specular_level is None else semantics.specular_level.astype(np.float32),
        metadata={
            'depth': depth.metadata,
            'semantics': semantics.metadata,
            'protected_region_fraction': float(np.mean(semantics.protected_mask > 0.05)) if semantics.protected_mask is not None else None,
            'semantic_label_keys': sorted(semantics.class_masks.keys()),
            'instance_mask_keys': sorted(semantics.instance_masks.keys()),
            'semantic_routing_is_domain_agnostic': True,
            'alpha_preserved': alpha_mask is not None,
            'material_maps_estimated': bool(semantics.roughness is not None or semantics.metalness is not None or semantics.specular_level is not None),
            'neural_scene_estimation_used': bool(depth.metadata.get('model_id') or semantics.metadata.get('detector_id') or semantics.metadata.get('model_id')),
        }
    )
