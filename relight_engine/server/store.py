"""Thread-safe LRU scene cache, disk persistence, and preview downscaling."""
from __future__ import annotations

from collections import OrderedDict
from copy import copy
from dataclasses import dataclass
from pathlib import Path
import re
import threading
import cv2
import numpy as np

from ..contracts import CameraModel, SceneBuffers
from ..workflows import load_scene_artifacts

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SAFE_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.(png|jpg|jpeg|json)$")


def validate_scene_id(scene_id: str) -> str:
    if not SAFE_ID_RE.match(scene_id):
        raise ValueError(f"Invalid identifier {scene_id!r}.")
    return scene_id


def validate_artifact_name(filename: str) -> str:
    if not SAFE_ARTIFACT_RE.match(filename) or ".." in filename:
        raise ValueError(f"Invalid artifact filename {filename!r}.")
    return filename


@dataclass
class CachedScene:
    scene_id: str
    scene_dir: Path
    scene: SceneBuffers
    camera: CameraModel
    manifest: dict
    arrays: dict[str, np.ndarray]


def _resize_map(arr: np.ndarray, target_wh: tuple[int, int], *, is_mask: bool = False) -> np.ndarray:
    tw, th = target_wh
    interp = cv2.INTER_LINEAR if is_mask else cv2.INTER_AREA
    out = cv2.resize(arr.astype(np.float32), (tw, th), interpolation=interp)
    if arr.ndim == 2 and out.ndim == 3:
        out = out[..., 0]
    return out.astype(np.float32)


def downsample_scene_for_preview(
    scene: SceneBuffers,
    camera: CameraModel,
    arrays: dict[str, np.ndarray],
    max_side: int,
) -> tuple[SceneBuffers, CameraModel, dict[str, np.ndarray]]:
    """Proportionally downscale scene buffers and pinhole intrinsics for fast interactive preview."""
    h, w = scene.shape
    longest = max(h, w)
    if longest <= max_side:
        return copy(scene), camera, arrays

    scale = float(max_side) / float(longest)
    tw = max(4, int(round(w * scale)))
    th = max(4, int(round(h * scale)))
    sx = float(tw) / float(w)
    sy = float(th) / float(h)

    scaled_camera = CameraModel(
        fx=camera.fx * sx,
        fy=camera.fy * sy,
        cx=camera.cx * sx,
        cy=camera.cy * sy,
    )
    scaled_normals = None
    if scene.normals is not None:
        nrm = _resize_map(scene.normals, (tw, th))
        lens = np.linalg.norm(nrm, axis=-1, keepdims=True)
        valid = lens[..., 0] > 1e-6
        nrm = np.where(valid[..., None], nrm / np.maximum(lens, 1e-6), np.array([0.0, 0.0, -1.0], np.float32))
        scaled_normals = nrm.astype(np.float32)

    scaled_arrays: dict[str, np.ndarray] = {}
    for key, val in arrays.items():
        if key.startswith("mask_") or key in ("lightable_mask", "protected_mask", "alpha_mask"):
            scaled_arrays[key] = np.clip(_resize_map(val, (tw, th), is_mask=True), 0.0, 1.0)
        elif key == "normals" and scaled_normals is not None:
            scaled_arrays[key] = scaled_normals
        else:
            scaled_arrays[key] = _resize_map(val, (tw, th))

    def _opt(arr: np.ndarray | None, *, mask: bool = False, clip01: bool = False) -> np.ndarray | None:
        if arr is None:
            return None
        resized = _resize_map(arr, (tw, th), is_mask=mask)
        return np.clip(resized, 0.0, 1.0).astype(np.float32) if clip01 else resized

    scaled_scene = SceneBuffers(
        rgb_srgb=np.clip(_resize_map(scene.rgb_srgb, (tw, th)), 0.0, 1.0),
        depth=np.maximum(_resize_map(scene.depth, (tw, th)), 0.0),
        depth_confidence=np.clip(_resize_map(scene.depth_confidence, (tw, th), is_mask=True), 0.0, 1.0),
        lightable_mask=np.clip(_resize_map(scene.lightable_mask, (tw, th), is_mask=True), 0.0, 1.0),
        depth_kind=scene.depth_kind,
        roughness=_opt(scene.roughness, clip01=True),
        metalness=_opt(scene.metalness, clip01=True),
        specular_level=_opt(scene.specular_level, clip01=True),
        clearcoat=_opt(scene.clearcoat, clip01=True),
        clearcoat_roughness=_opt(scene.clearcoat_roughness, clip01=True),
        subsurface=_opt(scene.subsurface, clip01=True),
        anisotropy=_opt(scene.anisotropy, clip01=True),
        anisotropy_angle=_opt(scene.anisotropy_angle),
        normals=scaled_normals,
        geometry_confidence=_opt(scene.geometry_confidence, mask=True, clip01=True),
        alpha_mask=_opt(scene.alpha_mask, mask=True, clip01=True),
        metadata=dict(scene.metadata),
        protected_mask=_opt(scene.protected_mask, mask=True, clip01=True),
        semantic_masks={k: np.clip(_resize_map(v, (tw, th), is_mask=True), 0.0, 1.0) for k, v in scene.semantic_masks.items()},
        instance_masks={k: np.clip(_resize_map(v, (tw, th), is_mask=True), 0.0, 1.0) for k, v in scene.instance_masks.items()},
        specular_color=_opt(scene.specular_color, clip01=True),
        material_confidence=_opt(scene.material_confidence, mask=True, clip01=True),
        material_metadata=dict(scene.material_metadata),
        albedo=_opt(scene.albedo, clip01=True),
        source_shading=None if scene.source_shading is None else np.maximum(_resize_map(scene.source_shading, (tw, th)), 0.0),
        non_diffuse_residual=None if scene.non_diffuse_residual is None else np.maximum(_resize_map(scene.non_diffuse_residual, (tw, th)), 0.0),
        intrinsic_metadata=dict(scene.intrinsic_metadata),
    )
    scaled_scene.validate()
    return scaled_scene, scaled_camera, scaled_arrays


class SceneStore:
    """Disk-backed scene & render repository with an LRU memory cache."""

    def __init__(self, root_dir: str | Path, *, max_cached_scenes: int = 8) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.scenes_dir = self.root_dir / "scenes"
        self.renders_dir = self.root_dir / "renders"
        self.uploads_dir = self.root_dir / "uploads"
        for d in (self.scenes_dir, self.renders_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.max_cached_scenes = max(1, int(max_cached_scenes))
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, CachedScene] = OrderedDict()
        self._external_dirs: dict[str, Path] = {}

    def register_preload_dir(self, base_dir: str | Path) -> list[str]:
        """Register any existing analyzed scene directories under base_dir."""
        base = Path(base_dir).resolve()
        loaded: list[str] = []
        if not base.is_dir():
            return loaded
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "scene_manifest.json").is_file() and (child / "scene_buffers.npz").is_file():
                try:
                    sid = validate_scene_id(child.name)
                except ValueError:
                    continue
                with self._lock:
                    self._external_dirs[sid] = child
                loaded.append(sid)
        return loaded

    def scene_dir_for(self, scene_id: str) -> Path:
        sid = validate_scene_id(scene_id)
        with self._lock:
            if sid in self._external_dirs:
                return self._external_dirs[sid]
        return self.scenes_dir / sid

    def list_scene_ids(self) -> list[str]:
        with self._lock:
            ids = set(self._external_dirs.keys())
        if self.scenes_dir.is_dir():
            for child in self.scenes_dir.iterdir():
                if child.is_dir() and (child / "scene_manifest.json").is_file():
                    ids.add(child.name)
        return sorted(ids)

    def cached_count(self) -> int:
        with self._lock:
            return len(self._cache)

    def get_scene(self, scene_id: str) -> CachedScene:
        sid = validate_scene_id(scene_id)
        with self._lock:
            if sid in self._cache:
                self._cache.move_to_end(sid)
                return self._cache[sid]
            sdir = self.scene_dir_for(sid)

        if not (sdir / "scene_manifest.json").is_file() or not (sdir / "scene_buffers.npz").is_file():
            raise FileNotFoundError(f"Scene {sid!r} not found.")

        scene, camera, manifest, arrays = load_scene_artifacts(sdir)
        entry = CachedScene(sid, sdir, scene, camera, manifest, arrays)
        with self._lock:
            self._cache[sid] = entry
            self._cache.move_to_end(sid)
            while len(self._cache) > self.max_cached_scenes:
                self._cache.popitem(last=False)
        return entry

    def invalidate(self, scene_id: str) -> None:
        sid = validate_scene_id(scene_id)
        with self._lock:
            self._cache.pop(sid, None)
