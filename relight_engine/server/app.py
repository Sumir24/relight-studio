"""FastAPI application factory for Relight Studio backend."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace
import hashlib
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
from ..geometry import place_point_light_at_uv, place_studio_light
from ..image_io import ALPHA_CONVENTION, save_image
from ..workflows import (
    apply_user_material_assignments,
    source_calibration_from_dict,
    source_illumination_from_dict,
    target_rig_from_dict,
)
from .schemas import (
    AnalyzeResponse,
    CancelRelightRequest,
    CancelRelightResponse,
    HealthResponse,
    JobStatusResponse,
    RelightRequest,
    RelightResponse,
    SceneSummaryResponse,
)
from .store import (
    SceneStore,
    atomic_write_bytes,
    atomic_write_text,
    downsample_scene_for_preview,
    validate_artifact_name,
    validate_scene_id,
)
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
        upload_sha = hashlib.sha256(raw).hexdigest()
        upload_path = store.uploads_dir / f"{sid}__{upload_sha[:12]}{ext}"
        if not upload_path.exists():
            atomic_write_bytes(upload_path, raw)

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

    @app.get("/v1/jobs", response_model=list[JobStatusResponse], dependencies=[Depends(require_auth)])
    def list_jobs(scene_id: str | None = Query(default=None)) -> list[JobStatusResponse]:
        try:
            jobs = queue.list_jobs(scene_id=scene_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return [
            JobStatusResponse(
                job_id=j.job_id,
                scene_id=j.scene_id,
                status=j.status,  # type: ignore[arg-type]
                stage=j.stage,
                progress=j.progress,
                error=j.error,
                scene_url=f"/v1/scenes/{j.scene_id}" if j.status == "ready" else None,
            )
            for j in jobs
        ]

    @app.post("/v1/scenes/{scene_id}/relight/cancel", response_model=CancelRelightResponse, dependencies=[Depends(require_auth)])
    def cancel_relight(scene_id: str, req: CancelRelightRequest) -> CancelRelightResponse:
        try:
            nxt = store.cancel_revisions(scene_id, req.client_session_id, req.min_revision_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CancelRelightResponse(
            scene_id=validate_scene_id(scene_id),
            client_session_id=req.client_session_id,
            cancelled_below_revision=nxt,
        )

    @app.post("/v1/scenes/{scene_id}/relight", dependencies=[Depends(require_auth)])
    def relight_scene(
        scene_id: str,
        req: RelightRequest,
        raw: bool = Query(default=False, description="Return image/png bytes directly instead of JSON metadata."),
    ) -> Response:
        started = perf_counter()
        client_sid = req.client_session_id or "default"
        if req.revision_id is not None:
            try:
                if not store.register_revision(scene_id, client_sid, req.revision_id):
                    raise HTTPException(status_code=409, detail=f"Stale relight revision {req.revision_id} superseded.")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            cached = store.get_scene(scene_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        session_hit = False
        try:
            placement_keys = {"placement_mode", "uv", "azimuth_deg", "elevation_deg", "distance_scale"}
            if req.source_illumination is not None:
                illum_data = req.source_illumination.model_dump(exclude_none=True)
                src_lights = []
                for l in req.source_illumination.lights:
                    ld = l.model_dump(exclude_none=True)
                    for pk in placement_keys:
                        ld.pop(pk, None)
                    src_lights.append(ld)
                illum_data["lights"] = src_lights
                source_rig = source_illumination_from_dict(illum_data).to_rig()
            elif req.source_calibration is not None:
                source_rig = source_calibration_from_dict(req.source_calibration.model_dump()).to_rig()
            else:
                raise ValueError("Provide either source_calibration or source_illumination.")
            target_rig_data = req.target_rig.model_dump(exclude_none=True)
            resolved_lights = []
            for l_schema, l_dict in zip(req.target_rig.lights, [l.model_dump(exclude_none=True) for l in req.target_rig.lights]):
                for pk in placement_keys:
                    l_dict.pop(pk, None)
                if l_schema.placement_mode == "image_uv":
                    u_v = l_schema.uv if l_schema.uv is not None else (0.25, 0.25)
                    l_dict["position"] = list(place_point_light_at_uv(cached.scene, cached.camera, float(u_v[0]), float(u_v[1])))
                elif l_schema.placement_mode == "studio_orbit":
                    pos_o, aim_o = place_studio_light(
                        cached.scene,
                        cached.camera,
                        azimuth_deg=float(l_schema.azimuth_deg),
                        elevation_deg=float(l_schema.elevation_deg),
                        distance_scale=float(l_schema.distance_scale),
                    )
                    l_dict["position"] = list(pos_o)
                    if "direction" not in l_dict and l_schema.kind in ("spot", "area", "point"):
                        l_dict["direction"] = list(aim_o)
                resolved_lights.append(l_dict)
            target_rig_data["lights"] = resolved_lights
            target_rig = target_rig_from_dict(target_rig_data)

            explicit_fields = set(getattr(req.settings, "model_fields_set", set())) - {"preset"}
            s_dump = req.settings.model_dump()
            preset_name = s_dump.pop("preset", None)
            overrides = {k: s_dump[k] for k in explicit_fields if k in s_dump}
            if preset_name == "studio_3d":
                settings = EngineSettings.studio_3d(**overrides)
            elif preset_name == "real_preserving":
                settings = EngineSettings.real_preserving(**overrides)
            else:
                settings = EngineSettings(**s_dump)
            settings.validate()
            material_dicts = [m.model_dump(exclude_none=True) for m in req.material_assignments]

            if settings.renderer_mode == "hybrid_neural":
                if not req.relighter_backend:
                    raise ValueError("renderer_mode='hybrid_neural' requires relighter_backend.")
                if req.preview_max_side is not None:
                    work_scene, work_cam, work_arrays = downsample_scene_for_preview(
                        cached.scene, cached.camera, cached.arrays, req.preview_max_side
                    )
                else:
                    work_scene = copy(cached.scene)
                    work_cam = cached.camera
                    work_arrays = cached.arrays
                material_meta = apply_user_material_assignments(work_scene, work_arrays, material_dicts)
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
                    out_alpha = work_scene.alpha_mask
                finally:
                    relighter.close()
            else:
                sess_preview_side = req.preview_max_side if req.preview_max_side is not None else (settings.preview_max_side or 384)
                render_settings = (
                    replace(settings, preview_max_side=req.preview_max_side, preview_upsample=False)
                    if req.preview_max_side is not None
                    else settings
                )
                entry, session_hit = store.get_or_create_relight_session(
                    scene_id,
                    source_rig,
                    material_dicts,
                    settings,
                    preview_max_side=sess_preview_side,
                    require_geometry_ready=req.require_geometry_ready,
                    apply_materials_fn=apply_user_material_assignments,
                )
                material_meta = entry.material_meta
                result = entry.session.render(target_rig, render_settings)
                out_alpha = (
                    entry.session.prev_scene.alpha_mask
                    if result.rgb_srgb.shape[:2] == entry.session.prev_scene.shape
                    else entry.session.work_scene.alpha_mask
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if req.revision_id is not None and store.is_revision_stale(scene_id, client_sid, req.revision_id):
            raise HTTPException(status_code=409, detail=f"Relight revision {req.revision_id} cancelled or superseded.")

        elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
        render_id = f"ren_{uuid.uuid4().hex[:12]}"
        out_h, out_w = result.rgb_srgb.shape[:2]
        persist = req.persist if req.persist is not None else not (raw and req.preview_max_side is not None)

        if persist:
            render_path = store.renders_dir / f"{render_id}.png"
            tmp_render = store.renders_dir / f".{render_id}.{uuid.uuid4().hex[:6]}.png"
            report_path = store.renders_dir / f"{render_id}.json"
            save_image(tmp_render, result.rgb_srgb, alpha_mask=out_alpha, bit_depth=req.bit_depth)
            tmp_render.replace(render_path)

            req_digest = hashlib.sha256(
                json.dumps(req.model_dump(mode="json"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            report: dict[str, Any] = {
                "render_report_version": "2.0",
                "scene_schema_version": cached.scene.schema_version,
                "scene_id": cached.scene_id,
                "render_id": render_id,
                "elapsed_ms": elapsed_ms,
                "resolution": [int(out_w), int(out_h)],
                "bit_depth": int(req.bit_depth),
                "alpha_convention": ALPHA_CONVENTION,
                "session_cache_hit": bool(session_hit),
                "revision_id": req.revision_id,
                "request_sha256": req_digest,
                "material_assignments": material_meta,
                "engine_metrics": result.metrics,
            }
            atomic_write_text(report_path, json.dumps(report, indent=2))

        if raw:
            if persist:
                png_bytes = render_path.read_bytes()
            else:
                rgb_u8 = np.round(np.clip(result.rgb_srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                buf = io.BytesIO()
                if out_alpha is not None:
                    a_u8 = np.round(np.clip(out_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
                    Image.fromarray(np.dstack([rgb_u8, a_u8]), mode="RGBA").save(buf, format="PNG")
                else:
                    Image.fromarray(rgb_u8, mode="RGB").save(buf, format="PNG")
                png_bytes = buf.getvalue()
            return Response(
                content=png_bytes,
                media_type="image/png",
                headers={
                    "X-Render-Id": render_id,
                    "X-Elapsed-Ms": str(elapsed_ms),
                    "X-Render-Persisted": "true" if persist else "false",
                    "X-Session-Cache-Hit": "true" if session_hit else "false",
                },
            )

        payload = RelightResponse(
            scene_id=cached.scene_id,
            render_id=render_id,
            render_url=f"/v1/renders/{render_id}.png" if persist else "",
            report_url=f"/v1/renders/{render_id}.json" if persist else "",
            elapsed_ms=elapsed_ms,
            resolution=[int(out_w), int(out_h)],
            metrics={**result.metrics, "session_cache_hit": bool(session_hit), "persisted": bool(persist)},
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
