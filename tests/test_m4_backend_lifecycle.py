"""Milestone M4 (Phases 7-8) tests: RelightSession caching, transient previews, revision cancellation, durable jobs, and reproducible reports."""
from __future__ import annotations

import json
from pathlib import Path
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from relight_engine.server import ServerConfig, create_app
from tests.test_server import _png_bytes, _write_synthetic_scene


def test_phase7_session_cache_transient_previews_and_revision_cancellation(tmp_path):
    preload_root = tmp_path / "preloaded"
    _write_synthetic_scene(preload_root / "sess_scene", h=40, w=60)
    storage = tmp_path / "srv"
    app = create_app(ServerConfig(storage_dir=storage, device="cpu", preload_dirs=(preload_root,)))

    base_req = {
        "source_illumination": {"lights": [{"kind": "directional", "direction": [0.0, 0.2, -0.97], "strength": 0.9}]},
        "target_rig": {
            "ambient_strength": 0.15,
            "lights": [{"kind": "point", "position": [-0.6, 0.5, 0.5], "strength": 2.0, "color": [1.0, 0.8, 0.6]}],
        },
        "client_session_id": "ui_tab_1",
        "revision_id": 10,
    }

    with TestClient(app) as client:
        # 1. Transient raw preview avoids creating disk render artifacts
        r1 = client.post("/v1/scenes/sess_scene/relight?raw=true", json={**base_req, "preview_max_side": 32, "revision_id": 10})
        assert r1.status_code == 200
        assert r1.headers["X-Render-Persisted"] == "false"
        assert r1.headers["X-Session-Cache-Hit"] == "false"
        assert list((storage / "renders").iterdir()) == []

        # 2. Changing light target on same preview_max_side hits cached RelightSession cleanly
        req2 = {
            **base_req,
            "preview_max_side": 32,
            "revision_id": 11,
            "target_rig": {
                "ambient_strength": 0.15,
                "lights": [{"kind": "point", "position": [0.7, 0.4, 0.6], "strength": 2.4, "color": [0.4, 0.7, 1.0]}],
            },
        }
        r2 = client.post("/v1/scenes/sess_scene/relight?raw=true", json=req2)
        assert r2.status_code == 200
        assert r2.headers["X-Session-Cache-Hit"] == "true"
        assert list((storage / "renders").iterdir()) == []

        # 3. Stale revision returns 409 Conflict immediately
        stale = client.post("/v1/scenes/sess_scene/relight?raw=true", json={**base_req, "preview_max_side": 32, "revision_id": 9})
        assert stale.status_code == 409

        # 4. Explicit cancel advances revision watermark
        c = client.post("/v1/scenes/sess_scene/relight/cancel", json={"client_session_id": "ui_tab_1", "min_revision_id": 20})
        assert c.status_code == 200
        assert c.json()["cancelled_below_revision"] == 20
        superseded = client.post("/v1/scenes/sess_scene/relight", json={**base_req, "revision_id": 15})
        assert superseded.status_code == 409

        # 5. Final render at higher revision persists PNG + reproducible JSON report
        final_res = client.post("/v1/scenes/sess_scene/relight", json={**base_req, "revision_id": 21, "bit_depth": 16})
        assert final_res.status_code == 200
        body = final_res.json()
        assert body["metrics"]["persisted"] is True
        rep = client.get(body["report_url"]).json()
        assert rep["render_report_version"] == "2.0"
        assert rep["bit_depth"] == 16
        assert rep["alpha_convention"] == "straight"
        assert rep["revision_id"] == 21
        assert len(rep["request_sha256"]) == 64


def test_phase8_durable_job_store_survives_restart_and_immutable_uploads(tmp_path):
    storage = tmp_path / "srv_durable"

    def fake_analyzer(image_path: Path, out_dir: Path, options: dict, device: str, on_progress):
        on_progress("estimating_depth", 0.5)
        _write_synthetic_scene(out_dir, h=24, w=32)

    app1 = create_app(ServerConfig(storage_dir=storage, device="cpu", analyzer_fn=fake_analyzer))
    with TestClient(app1) as c1:
        png = _png_bytes(32, 24)
        up1 = c1.post("/v1/scenes/analyze", files={"file": ("photo.png", png, "image/png")}, data={"scene_id": "job_scene"})
        assert up1.status_code == 202
        job_id = up1.json()["job_id"]
        for _ in range(40):
            js = c1.get(f"/v1/jobs/{job_id}").json()
            if js["status"] == "ready":
                break
            time.sleep(0.02)
        assert js["status"] == "ready"
        upload_files = list((storage / "uploads").iterdir())
        assert len(upload_files) == 1
        assert "__" in upload_files[0].name  # content-hashed immutable filename

    # Inject a crashed/orphaned queued job JSON on disk and restart server
    orphan_path = storage / "jobs" / "job_orphaned001.json"
    orphan_path.write_text(json.dumps({
        "job_id": "job_orphaned001",
        "scene_id": "job_scene",
        "status": "running",
        "stage": "estimating_depth",
        "progress": 0.4,
    }), encoding="utf-8")

    app2 = create_app(ServerConfig(storage_dir=storage, device="cpu"))
    with TestClient(app2) as c2:
        # Persisted ready job still visible after server restart
        res_ready = c2.get(f"/v1/jobs/{job_id}")
        assert res_ready.status_code == 200
        assert res_ready.json()["status"] == "ready"

        # Orphaned in-flight job cleanly recovered as failed across restart
        res_orphan = c2.get("/v1/jobs/job_orphaned001")
        assert res_orphan.status_code == 200
        assert res_orphan.json()["status"] == "failed"
        assert len(c2.get("/v1/jobs?scene_id=job_scene").json()) == 2
