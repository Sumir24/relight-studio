"""Sequential background worker for GPU/MPS scene analysis jobs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import threading
from typing import Callable
import uuid

from .store import SceneStore, validate_scene_id

ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYZE_SCRIPT = ROOT_DIR / "scripts" / "run_scene_analysis.py"


@dataclass
class AnalysisJob:
    job_id: str
    scene_id: str
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None


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
    """Single-worker sequential queue preventing concurrent GPU/MPS model loads."""

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

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    def get_job(self, job_id: str) -> AnalysisJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, scene_id: str, image_path: Path, options: dict) -> AnalysisJob:
        sid = validate_scene_id(scene_id)
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = AnalysisJob(job_id=job_id, scene_id=sid, status="queued", stage="queued", progress=0.0)
        with self._lock:
            self._jobs[job_id] = job
        out_dir = self.store.scene_dir_for(sid)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._executor.submit(self._run_job, job, image_path, out_dir, options)
        return job

    def _run_job(self, job: AnalysisJob, image_path: Path, out_dir: Path, options: dict) -> None:
        def update_progress(stage: str, progress: float) -> None:
            with self._lock:
                job.stage = stage
                job.progress = progress

        with self._lock:
            job.status = "running"
            job.stage = "starting"
            job.progress = 0.05
        try:
            self.analyzer_fn(image_path, out_dir, options, self.device, update_progress)
            self.store.invalidate(job.scene_id)
            self.store.get_scene(job.scene_id)
            with self._lock:
                job.status = "ready"
                job.stage = "complete"
                job.progress = 1.0
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.stage = "failed"
                job.error = str(exc)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
