"""Sequential background worker for GPU/MPS scene analysis jobs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable
import uuid

from .store import SceneStore, atomic_write_text, validate_scene_id

ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYZE_SCRIPT = ROOT_DIR / "scripts" / "run_scene_analysis.py"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AnalysisJob:
    job_id: str
    scene_id: str
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    input_sha256: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


def default_scene_analyzer(
    image_path: Path,
    out_dir: Path,
    options: dict,
    device: str,
    on_progress: Callable[[str, float], None],
) -> None:
    """Invoke the canonical scene analysis pipeline in a clean subprocess to guarantee device cleanup."""
    cmd = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "--image", str(image_path),
        "--out", str(out_dir),
        "--device", device,
        "--hfov", str(options.get("hfov", 55.0)),
        "--hfov-source", str(options.get("hfov_source", "assumed")),
        "--near-is-high" if options.get("near_is_high", True) else "--far-is-high",
        "--scene-domain", str(options.get("scene_domain", "auto")),
        "--normal-backend", str(options.get("normal_backend", "marigold_normals")),
        "--normal-ensemble", str(options.get("normal_ensemble", 3)),
        "--intrinsic-backend", str(options.get("intrinsic_backend", "marigold_intrinsics_lighting")),
        "--intrinsic-ensemble", str(options.get("intrinsic_ensemble", 3)),
        "--material-backend", str(options.get("material_backend", "marigold_intrinsics_appearance")),
        "--material-ensemble", str(options.get("material_ensemble", 3)),
    ]
    if options.get("prompts"):
        cmd.extend(["--prompts", *[str(p) for p in options["prompts"]]])
    if options.get("protected_labels"):
        cmd.extend(["--protected-labels", *[str(p) for p in options["protected_labels"]]])
    if options.get("lightable_labels"):
        cmd.extend(["--lightable-labels", *[str(p) for p in options["lightable_labels"]]])

    on_progress("initializing_models", 0.1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    tail_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.strip()
        if text:
            tail_lines.append(text)
            tail_lines = tail_lines[-25:]
        if "[1/5]" in text:
            on_progress("estimating_depth", 0.2)
        elif "[2/5]" in text:
            on_progress("discovering_regions", 0.35)
        elif "[3/5]" in text:
            on_progress("segmenting_masks", 0.5)
        elif "[4/5]" in text or "[4b/5]" in text:
            on_progress("estimating_normals", 0.7)
        elif "[4c/5]" in text:
            on_progress("estimating_intrinsics", 0.80)
        elif "[4d/5]" in text:
            on_progress("estimating_materials", 0.90)
        elif "[5/5]" in text:
            on_progress("writing_scene_buffers", 0.95)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Scene analysis failed (exit {rc}): {' | '.join(tail_lines[-5:])}")


class AnalysisQueue:
    """Single-worker sequential queue preventing concurrent GPU/MPS model loads, with durable job manifests."""

    def __init__(
        self,
        store: SceneStore,
        *,
        device: str = "auto",
        analyzer_fn: Callable[[Path, Path, dict, str, Callable[[str, float], None]], None] | None = None,
    ) -> None:
        self.store = store
        self.device = device
        self.analyzer_fn = analyzer_fn or default_scene_analyzer
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="relight-analyzer")
        self._lock = threading.RLock()
        self._jobs: dict[str, AnalysisJob] = {}
        self._load_persisted_jobs()

    def _job_path(self, job_id: str) -> Path:
        return self.store.jobs_dir / f"{validate_scene_id(job_id)}.json"

    def _persist_job(self, job: AnalysisJob) -> None:
        job.updated_at = _utc_now_iso()
        atomic_write_text(self._job_path(job.job_id), json.dumps(asdict(job), indent=2))

    def _load_persisted_jobs(self) -> None:
        if not self.store.jobs_dir.is_dir():
            return
        for path in sorted(self.store.jobs_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                job = AnalysisJob(
                    job_id=str(raw["job_id"]),
                    scene_id=str(raw["scene_id"]),
                    status=str(raw.get("status", "failed")),
                    stage=str(raw.get("stage", "failed")),
                    progress=float(raw.get("progress", 0.0)),
                    error=raw.get("error"),
                    created_at=str(raw.get("created_at", _utc_now_iso())),
                    updated_at=str(raw.get("updated_at", _utc_now_iso())),
                    input_sha256=raw.get("input_sha256"),
                    options=dict(raw.get("options", {})),
                )
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    job.stage = "interrupted"
                    job.error = "Analysis interrupted by server restart before completion."
                    self._persist_job(job)
                self._jobs[job.job_id] = job
            except Exception:
                continue

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    def list_jobs(self, scene_id: str | None = None) -> list[AnalysisJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        if scene_id is not None:
            sid = validate_scene_id(scene_id)
            jobs = [j for j in jobs if j.scene_id == sid]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def get_job(self, job_id: str) -> AnalysisJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, scene_id: str, image_path: Path, options: dict) -> AnalysisJob:
        sid = validate_scene_id(scene_id)
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        input_sha = hashlib.sha256(image_path.read_bytes()).hexdigest() if image_path.is_file() else None
        job = AnalysisJob(
            job_id=job_id,
            scene_id=sid,
            status="queued",
            stage="queued",
            progress=0.0,
            input_sha256=input_sha,
            options=dict(options),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._persist_job(job)
        out_dir = self.store.scene_dir_for(sid)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._executor.submit(self._run_job, job, image_path, out_dir, options)
        return job

    def _run_job(self, job: AnalysisJob, image_path: Path, out_dir: Path, options: dict) -> None:
        def update_progress(stage: str, progress: float) -> None:
            with self._lock:
                job.stage = stage
                job.progress = progress
                self._persist_job(job)

        with self._lock:
            job.status = "running"
            job.stage = "starting"
            job.progress = 0.05
            self._persist_job(job)
        try:
            self.analyzer_fn(image_path, out_dir, options, self.device, update_progress)
            self.store.invalidate(job.scene_id)
            self.store.get_scene(job.scene_id)
            with self._lock:
                job.status = "ready"
                job.stage = "complete"
                job.progress = 1.0
                self._persist_job(job)
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.stage = "failed"
                job.error = str(exc)
                self._persist_job(job)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
