from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from relight_engine.server import ServerConfig, create_app


def _write_synthetic_scene(scene_dir: Path, h: int = 32, w: int = 48) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([
        np.full((h, w), 0.45, np.float32),
        0.35 + 0.10 * (xx / max(w - 1, 1)),
        0.32 + 0.08 * (yy / max(h - 1, 1)),
    ], axis=-1).astype(np.float32)
    depth = np.where(xx < w // 2, 1.5, 3.0).astype(np.float32)
    conf = np.ones((h, w), np.float32)
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = -1.0
    lightable = np.ones((h, w), np.float32)
    subject = np.zeros((h, w), np.float32)
    subject[:, : w // 2] = 1.0

    np.savez_compressed(
        scene_dir / "scene_buffers.npz",
        rgb_srgb=rgb,
        depth=depth,
        depth_confidence=conf,
        geometry_confidence=conf,
        normals=normals,
        lightable_mask=lightable,
        mask_subject=subject,
    )
    Image.fromarray(np.round(rgb * 255).astype(np.uint8), mode="RGB").save(scene_dir / "semantic_overlay.jpg")
    manifest = {
        "resolution": [w, h],
        "depth": {"depth_kind": "relative"},
        "semantic_labels": ["subject"],
        "instance_masks": [],
        "camera": {"fx": 45.0, "fy": 45.0, "cx": w / 2, "cy": h / 2},
        "metrics": {"lightable_fraction": 1.0, "geometry_confidence_mean": 1.0},
    }
    (scene_dir / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _png_bytes(w: int = 48, h: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(120, 140, 160)).save(buf, format="PNG")
    return buf.getvalue()


def test_server_analyze_job_lifecycle_and_relight_endpoints(tmp_path):
    def fake_analyzer(image_path: Path, out_dir: Path, options: dict, device: str, on_progress):
        assert image_path.is_file()
        assert device == "cpu"
        on_progress("estimating_depth", 0.3)
        _write_synthetic_scene(out_dir, h=40, w=60)
        on_progress("complete", 1.0)

    app = create_app(ServerConfig(
        storage_dir=tmp_path / "srv",
        device="cpu",
        analyzer_fn=fake_analyzer,
    ))
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        upload = client.post(
            "/v1/scenes/analyze",
            files={"file": ("photo.png", _png_bytes(60, 40), "image/png")},
            data={"scene_id": "demo_scene", "hfov": "55", "prompts": "subject"},
        )
        assert upload.status_code == 202
        job_id = upload.json()["job_id"]

        for _ in range(40):
            js = client.get(f"/v1/jobs/{job_id}").json()
            if js["status"] == "ready":
                break
            time.sleep(0.02)
        assert js["status"] == "ready"
        assert "demo_scene" in client.get("/v1/scenes").json()

        summary = client.get("/v1/scenes/demo_scene")
        assert summary.status_code == 200
        assert summary.json()["resolution"] == [60, 40]
        assert "semantic_overlay.jpg" in summary.json()["artifacts"]

        art = client.get("/v1/scenes/demo_scene/artifacts/semantic_overlay.jpg")
        assert art.status_code == 200
        assert len(art.content) > 100

        relight_payload = {
            "source_calibration": {"key_direction": [0.0, 0.5, -0.86], "key_strength": 0.9},
            "target_rig": {
                "ambient_strength": 0.2,
                "lights": [
                    {
                        "name": "warm_key",
                        "kind": "point",
                        "position": [-0.8, 0.6, 0.4],
                        "color": [1.0, 0.72, 0.40],
                        "strength": 2.4,
                        "radius": 0.25,
                    },
                    {
                        "name": "blue_rim",
                        "kind": "point",
                        "position": [0.8, 0.4, 0.8],
                        "color": [0.25, 0.55, 1.0],
                        "strength": 1.8,
                    },
                ],
            },
            "material_assignments": [
                {"name": "subject_coat", "mask_key": "subject", "roughness": 0.35, "clearcoat": 0.5}
            ],
            "settings": {
                "detail_power": 1.0,
                "indirect_bounce_strength": 0.2,
                "filmic_highlight_rolloff": True,
                "refine_depth_edges": True,
            },
        }
        res_full = client.post("/v1/scenes/demo_scene/relight", json=relight_payload)
        assert res_full.status_code == 200
        body = res_full.json()
        assert body["resolution"] == [60, 40]
        img_resp = client.get(body["render_url"])
        assert img_resp.status_code == 200
        assert img_resp.headers["content-type"] == "image/png"

        # Interactive preview downscaling + raw PNG response
        preview_payload = {**relight_payload, "preview_max_side": 32}
        res_raw = client.post("/v1/scenes/demo_scene/relight?raw=true", json=preview_payload)
        assert res_raw.status_code == 200
        assert res_raw.headers["content-type"] == "image/png"
        with Image.open(io.BytesIO(res_raw.content)) as preview_img:
            assert max(preview_img.size) == 32


def test_server_guards_auth_and_rejects_bad_inputs(tmp_path):
    preload_root = tmp_path / "preloaded"
    _write_synthetic_scene(preload_root / "existing_scene")

    app = create_app(ServerConfig(
        storage_dir=tmp_path / "srv",
        device="cpu",
        preload_dirs=(preload_root,),
        max_upload_bytes=128,
        max_image_pixels=400,
        api_key="secret-token",
    ))
    with TestClient(app) as client:
        # Health remains open; protected endpoints require Bearer token
        assert client.get("/health").status_code == 200
        assert client.get("/v1/scenes").status_code == 401

        hdr = {"Authorization": "Bearer secret-token"}
        assert "existing_scene" in client.get("/v1/scenes", headers=hdr).json()

        # Reject path traversal / bad filenames
        assert client.get("/v1/scenes/..%2Fetc/artifacts/passwd.png", headers=hdr).status_code in (400, 404)
        assert client.get("/v1/scenes/existing_scene/artifacts/scene_buffers.npz", headers=hdr).status_code == 400

        # Reject oversized upload
        big_bytes = _png_bytes(100, 100)
        too_big = client.post(
            "/v1/scenes/analyze",
            headers=hdr,
            files={"file": ("big.png", big_bytes, "image/png")},
        )
        assert too_big.status_code == 413
