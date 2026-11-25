#!/usr/bin/env python3
"""Launch the Relight Studio FastAPI backend server."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight_engine.server import ServerConfig, create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the Relight Studio HTTP backend server.")
    parser.add_argument("--host", default=os.environ.get("RELIGHT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RELIGHT_PORT", "8000")))
    parser.add_argument("--storage-dir", type=Path, default=Path(os.environ.get("RELIGHT_STORAGE_DIR", "outputs/server_data")))
    parser.add_argument("--preload-dir", type=Path, action="append", default=[],
                        help="Existing analyzed scenes root directory to mount on startup.")
    parser.add_argument("--device", default=os.environ.get("RELIGHT_DEVICE", "auto"),
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-cached-scenes", type=int, default=8)
    parser.add_argument("--api-key", default=os.environ.get("RELIGHT_API_KEY", None))
    args = parser.parse_args()

    import uvicorn

    cfg = ServerConfig(
        storage_dir=args.storage_dir,
        device=args.device,
        preload_dirs=tuple(args.preload_dir),
        max_cached_scenes=args.max_cached_scenes,
        api_key=args.api_key or None,
    )
    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
