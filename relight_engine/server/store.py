"""Thread-safe LRU scene cache, disk persistence, and preview downscaling."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any
import numpy as np

from ..contracts import CameraModel, EngineSettings, LightingRig, SceneBuffers
from ..engine import RelightSession
from ..preview import downsample_scene_and_arrays_for_preview
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


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically via a temporary sibling file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write UTF-8 text atomically via a temporary sibling file and os.replace."""
    atomic_write_bytes(path, text.encode(encoding))


@dataclass
class CachedScene:
    scene_id: str
    scene_dir: Path
    scene: SceneBuffers
    camera: CameraModel
    manifest: dict
    arrays: dict[str, np.ndarray]


@dataclass
class CachedSessionEntry:
    session: RelightSession
    material_meta: dict[str, Any]


def downsample_scene_for_preview(
    scene: SceneBuffers,
    camera: CameraModel,
    arrays: dict[str, np.ndarray],
    max_side: int,
) -> tuple[SceneBuffers, CameraModel, dict[str, np.ndarray]]:
    """Delegate preview downscaling to the single shared preview implementation."""
    return downsample_scene_and_arrays_for_preview(scene, camera, max_side, arrays=arrays)


class SceneStore:
    """Disk-backed scene, job, and render repository with LRU scene & RelightSession caches."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        max_cached_scenes: int = 8,
        max_cached_sessions: int = 16,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.scenes_dir = self.root_dir / "scenes"
        self.renders_dir = self.root_dir / "renders"
        self.uploads_dir = self.root_dir / "uploads"
        self.jobs_dir = self.root_dir / "jobs"
        for d in (self.scenes_dir, self.renders_dir, self.uploads_dir, self.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.max_cached_scenes = max(1, int(max_cached_scenes))
        self.max_cached_sessions = max(1, int(max_cached_sessions))
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, CachedScene] = OrderedDict()
        self._sessions: OrderedDict[tuple[str, str], CachedSessionEntry] = OrderedDict()
        self._rev_watermarks: dict[tuple[str, str], int] = {}
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

    def session_cache_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _session_signature(
        self,
        source_rig: LightingRig,
        material_dicts: list[dict[str, Any]],
        settings: EngineSettings,
        preview_max_side: int,
        require_geometry_ready: bool,
    ) -> str:
        payload = {
            "source_rig": {
                "ambient_color": [float(c) for c in source_rig.ambient_color],
                "ambient_strength": float(source_rig.ambient_strength),
                "lights": [
                    {
                        "light_id": l.light_id,
                        "name": l.name,
                        "kind": l.kind,
                        "position": list(l.position) if l.position is not None else None,
                        "direction": list(l.direction) if l.direction is not None else None,
                        "color": list(l.color),
                        "strength": float(l.strength),
                        "radius": float(l.radius),
                        "area_size": list(l.area_size),
                        "samples": int(l.samples),
                        "casts_shadows": bool(l.casts_shadows),
                        "spot_inner_angle_deg": float(l.spot_inner_angle_deg),
                        "spot_outer_angle_deg": float(l.spot_outer_angle_deg),
                        "intensity_mode": str(l.intensity_mode),
                    }
                    for l in source_rig.lights
                ],
                "environment": (
                    None
                    if source_rig.environment is None
                    else {
                        "zenith_color": list(source_rig.environment.zenith_color),
                        "horizon_color": list(source_rig.environment.horizon_color),
                        "ground_color": list(source_rig.environment.ground_color),
                        "strength": float(source_rig.environment.strength),
                        "sh_coeffs": None if source_rig.environment.sh_coeffs is None else np.asarray(source_rig.environment.sh_coeffs).tolist(),
                    }
                ),
            },
            "materials": material_dicts,
            "session_settings": {
                "refine_depth_edges": bool(settings.refine_depth_edges),
                "normal_smooth_sigma": float(settings.normal_smooth_sigma),
                "min_geometry_confidence": float(settings.min_geometry_confidence),
                "detail_preservation_mode": str(settings.detail_preservation_mode),
                "detail_lock_strength": float(settings.detail_lock_strength),
                "base_blur_sigma": float(settings.base_blur_sigma),
                "edge_aware_base": bool(settings.edge_aware_base),
                "renderer_mode": "intrinsic" if settings.renderer_mode == "intrinsic" else "auto",
                "preview_max_side": int(preview_max_side),
                "require_geometry_ready": bool(require_geometry_ready),
            },
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]

    def get_or_create_relight_session(
        self,
        scene_id: str,
        source_rig: LightingRig,
        material_dicts: list[dict[str, Any]],
        settings: EngineSettings,
        *,
        preview_max_side: int = 384,
        require_geometry_ready: bool = False,
        apply_materials_fn: Any = None,
    ) -> tuple[CachedSessionEntry, bool]:
        """Return (CachedSessionEntry, cache_hit) keyed by scene, materials, source rig, and geometry settings."""
        sid = validate_scene_id(scene_id)
        sig = self._session_signature(source_rig, material_dicts, settings, preview_max_side, require_geometry_ready)
        key = (sid, sig)
        with self._lock:
            hit = self._sessions.get(key)
            if hit is not None:
                self._sessions.move_to_end(key)
                return hit, True

        cached = self.get_scene(sid)
        from copy import copy
        work_scene = copy(cached.scene)
        material_meta = (
            apply_materials_fn(work_scene, cached.arrays, material_dicts)
            if apply_materials_fn is not None
            else {}
        )
        session = RelightSession(
            work_scene,
            cached.camera,
            source_rig,
            settings,
            preview_max_side=preview_max_side,
            require_geometry_ready=require_geometry_ready,
        )
        entry = CachedSessionEntry(session=session, material_meta=material_meta)
        with self._lock:
            self._sessions[key] = entry
            self._sessions.move_to_end(key)
            while len(self._sessions) > self.max_cached_sessions:
                self._sessions.popitem(last=False)
        return entry, False

    def register_revision(self, scene_id: str, client_session_id: str, revision_id: int) -> bool:
        """Advance monotonic revision watermark; return False if revision is already stale."""
        sid = validate_scene_id(scene_id)
        k = (sid, str(client_session_id or "default"))
        with self._lock:
            cur = self._rev_watermarks.get(k, -1)
            if revision_id < cur:
                return False
            self._rev_watermarks[k] = int(revision_id)
            return True

    def is_revision_stale(self, scene_id: str, client_session_id: str, revision_id: int) -> bool:
        sid = validate_scene_id(scene_id)
        k = (sid, str(client_session_id or "default"))
        with self._lock:
            return int(revision_id) < self._rev_watermarks.get(k, -1)

    def cancel_revisions(self, scene_id: str, client_session_id: str, min_revision_id: int) -> int:
        sid = validate_scene_id(scene_id)
        k = (sid, str(client_session_id or "default"))
        with self._lock:
            cur = self._rev_watermarks.get(k, -1)
            nxt = max(cur, int(min_revision_id))
            self._rev_watermarks[k] = nxt
            return nxt

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
            for k in [key for key in self._sessions if key[0] == sid]:
                self._sessions.pop(k, None)
            for k in [key for key in self._rev_watermarks if key[0] == sid]:
                self._rev_watermarks.pop(k, None)
