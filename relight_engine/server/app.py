"""FastAPI application factory for Relight Studio backend."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import io
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
import numpy as np
from PIL import Image

from ..adapters.registry import available_backends, create_relighter_backend
from ..contracts import EngineSettings
from ..engine import relight
from ..image_io import save_image
from ..workflows import (
    apply_user_material_assignments,
    source_calibration_from_dict,
    target_rig_from_dict,
)
from .schemas import (
    AnalyzeResponse,
    HealthResponse,
    JobStatusResponse,
    RelightRequest,
    RelightResponse,
    SceneSummaryResponse,
)
from .store import SceneStore, downsample_scene_for_preview, validate_artifact_name, validate_scene_id
from .worker import AnalysisQueue


@dataclass
class ServerConfig:
    storage_dir: Path = Path("outputs/server_data")
    device: str = "auto"
    preload_dirs: tuple[Path, ...] = ()
    max_cached_scenes: int = 8
    max_upload_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 16_777_216
    api_key: str | None = None
    cors_origins: tuple[str, ...] = ("*",)
    analyzer_fn: Callable[[Path, Path, dict, str, Callable[[str, float], None]], None] | None = None


def _build_scene_summary(scene_id: str, scene_dir: Path, manifest: dict, scene_labels: list[str]) -> SceneSummaryResponse:
    artifacts: dict[str, str] = {}
    for candidate in (
        "depth_relative.png",
        "normals_camera_space.png",
        "geometry_confidence.png",
        "intrinsic_albedo.png",
        "intrinsic_shading.png",
        "mask_lightable.png",
        "mask_protected.png",
        "semantic_overlay.jpg",
    ):
        if (scene_dir / candidate).is_file():
            artifacts[candidate] = f"/v1/scenes/{scene_id}/artifacts/{candidate}"
    for path in sorted(scene_dir.glob("mask_*.png")):
        artifacts[path.name] = f"/v1/scenes/{scene_id}/artifacts/{path.name}"
    return SceneSummaryResponse(
        scene_id=scene_id,
        resolution=list(manifest.get("resolution", [0, 0])),
        depth_kind=str(manifest.get("depth", {}).get("depth_kind", "relative")),
        semantic_labels=sorted(set(manifest.get("semantic_labels", scene_labels))),
        instance_masks=sorted(manifest.get("instance_masks", [])),
        camera=dict(manifest.get("camera", {})),
        artifacts=artifacts,
        metrics=dict(manifest.get("metrics", {})),
    )


def create_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or ServerConfig()
    store = SceneStore(cfg.storage_dir, max_cached_scenes=cfg.max_cached_scenes)
    for preload in cfg.preload_dirs:
        store.register_preload_dir(preload)
    queue = AnalysisQueue(store, device=cfg.device, analyzer_fn=cfg.analyzer_fn)

    app = FastAPI(
        title="Relight Studio Backend API",
        version="1.0.0",
        description="Interactive 2.5D scene analysis and physics-based relighting service.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store
    app.state.queue = queue
    app.state.config = cfg

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not cfg.api_key:
            return
        expected = f"Bearer {cfg.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing Bearer API key.")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            device=cfg.device,
            cached_scenes=store.cached_count(),
            total_scenes=len(store.list_scene_ids()),
            active_jobs=queue.active_count(),
            available_backends=available_backends(),
        )

    @app.post("/v1/scenes/analyze", response_model=AnalyzeResponse, status_code=202, dependencies=[Depends(require_auth)])
    async def analyze_scene(
        file: UploadFile = File(...),
        scene_id: str | None = Form(default=None),
        hfov: float = Form(default=55.0),
        hfov_source: str = Form(default="assumed"),
        near_is_high: bool = Form(default=True),
        scene_domain: str = Form(default="auto"),
        normal_backend: str = Form(default="marigold_normals"),
        normal_ensemble: int = Form(default=3),
        intrinsic_backend: str = Form(default="marigold_intrinsics_lighting"),
        intrinsic_ensemble: int = Form(default=3),
        prompts: str | None = Form(default=None),
        protected_labels: str | None = Form(default=None),
        lightable_labels: str | None = Form(default=None),
    ) -> AnalyzeResponse:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
        if len(raw) > cfg.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"Image exceeds max upload size ({cfg.max_upload_bytes} bytes).")
        try:
            with Image.open(io.BytesIO(raw)) as img:
                w, h = img.size
                if w * h > cfg.max_image_pixels:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image resolution {w}x{h} exceeds max pixel guard ({cfg.max_image_pixels}).",
                    )
                img.verify()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc

        try:
            sid = validate_scene_id(scene_id or f"scene_{uuid.uuid4().hex[:12]}")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        ext = Path(file.filename or "upload.png").suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"):
            ext = ".png"
        upload_path = store.uploads_dir / f"{sid}{ext}"
        upload_path.write_bytes(raw)

        def _split_csv(value: str | None) -> list[str] | None:
            if not value:
                return None
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return parts or None

        options = {
            "hfov": float(hfov),
            "hfov_source": hfov_source,
            "near_is_high": bool(near_is_high),
            "scene_domain": scene_domain,
            "normal_backend": normal_backend,
            "normal_ensemble": max(1, int(normal_ensemble)),
            "intrinsic_backend": intrinsic_backend,
            "intrinsic_ensemble": max(1, int(intrinsic_ensemble)),
            "prompts": _split_csv(prompts),
            "protected_labels": _split_csv(protected_labels),
            "lightable_labels": _split_csv(lightable_labels),
        }
        job = queue.submit(sid, upload_path, options)
        return AnalyzeResponse(
            job_id=job.job_id,
            scene_id=sid,
            status=job.status,
            status_url=f"/v1/jobs/{job.job_id}",
            scene_url=f"/v1/scenes/{sid}",
        )

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_auth)])
    def get_job_status(job_id: str) -> JobStatusResponse:
        job = queue.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
        return JobStatusResponse(
            job_id=job.job_id,
            scene_id=job.scene_id,
            status=job.status,  # type: ignore[arg-type]
            stage=job.stage,
            progress=job.progress,
            error=job.error,
            scene_url=f"/v1/scenes/{job.scene_id}" if job.status == "ready" else None,
        )

    @app.get("/v1/scenes", response_model=list[str], dependencies=[Depends(require_auth)])
    def list_scenes() -> list[str]:
        return store.list_scene_ids()

    @app.get("/v1/scenes/{scene_id}", response_model=SceneSummaryResponse, dependencies=[Depends(require_auth)])
    def get_scene(scene_id: str) -> SceneSummaryResponse:
        try:
            cached = store.get_scene(scene_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _build_scene_summary(
            cached.scene_id,
            cached.scene_dir,
            cached.manifest,
            sorted(cached.scene.semantic_masks.keys()),
        )

    @app.get("/v1/scenes/{scene_id}/artifacts/{artifact_name}", dependencies=[Depends(require_auth)])
    def get_scene_artifact(scene_id: str, artifact_name: str) -> FileResponse:
        try:
            sid = validate_scene_id(scene_id)
            fname = validate_artifact_name(artifact_name)
            sdir = store.scene_dir_for(sid)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        path = (sdir / fname).resolve()
        if path.parent != sdir.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Artifact {fname!r} not found for scene {sid!r}.")
        return FileResponse(path)

    @app.post("/v1/scenes/{scene_id}/relight", dependencies=[Depends(require_auth)])
    def relight_scene(
        scene_id: str,
        req: RelightRequest,
        raw: bool = Query(default=False, description="Return image/png bytes directly instead of JSON metadata."),
    ) -> Response:
        started = perf_counter()
        try:
            cached = store.get_scene(scene_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            source_rig = source_calibration_from_dict(req.source_calibration.model_dump()).to_rig()
            target_rig_data = req.target_rig.model_dump(exclude_none=True)
            target_rig_data["lights"] = [l.model_dump(exclude_none=True) for l in req.target_rig.lights]
            target_rig = target_rig_from_dict(target_rig_data)
            settings = EngineSettings(**req.settings.model_dump())
            settings.validate()

            if req.preview_max_side is not None:
                work_scene, work_cam, work_arrays = downsample_scene_for_preview(
                    cached.scene, cached.camera, cached.arrays, req.preview_max_side
                )
            else:
                work_scene = copy(cached.scene)
                work_cam = cached.camera
                work_arrays = cached.arrays

            material_dicts = [m.model_dump(exclude_none=True) for m in req.material_assignments]
            material_meta = apply_user_material_assignments(work_scene, work_arrays, material_dicts)

            relighter = None
            if settings.renderer_mode == "hybrid_neural":
                if not req.relighter_backend:
                    raise ValueError("renderer_mode='hybrid_neural' requires relighter_backend.")
                opts = dict(req.relighter_options)
                if cfg.device != "auto" or "device" not in opts:
                    opts["device"] = cfg.device
                relighter = create_relighter_backend(req.relighter_backend, **opts)

            try:
                result = relight(
                    work_scene,
                    work_cam,
                    source_rig,
                    target_rig,
                    settings,
                    require_geometry_ready=req.require_geometry_ready,
                    neural_relighter=relighter,
                )
            finally:
                if relighter is not None:
                    relighter.close()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
        render_id = f"ren_{uuid.uuid4().hex[:12]}"
        render_path = store.renders_dir / f"{render_id}.png"
        report_path = store.renders_dir / f"{render_id}.json"

        save_image(render_path, result.rgb_srgb, alpha_mask=work_scene.alpha_mask)
        report: dict[str, Any] = {
            "scene_id": cached.scene_id,
            "render_id": render_id,
            "elapsed_ms": elapsed_ms,
            "resolution": [int(work_scene.shape[1]), int(work_scene.shape[0])],
            "material_assignments": material_meta,
            "engine_metrics": result.metrics,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        if raw:
            return Response(content=render_path.read_bytes(), media_type="image/png", headers={"X-Render-Id": render_id, "X-Elapsed-Ms": str(elapsed_ms)})

        payload = RelightResponse(
            scene_id=cached.scene_id,
            render_id=render_id,
            render_url=f"/v1/renders/{render_id}.png",
            report_url=f"/v1/renders/{render_id}.json",
            elapsed_ms=elapsed_ms,
            resolution=[int(work_scene.shape[1]), int(work_scene.shape[0])],
            metrics=result.metrics,
        )
        return JSONResponse(content=payload.model_dump())

    @app.get("/v1/renders/{filename}", dependencies=[Depends(require_auth)])
    def get_render_file(filename: str) -> FileResponse:
        try:
            fname = validate_artifact_name(filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        path = (store.renders_dir / fname).resolve()
        if path.parent != store.renders_dir.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Render artifact {fname!r} not found.")
        return FileResponse(path)

    return app
