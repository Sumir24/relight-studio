"""Shared runtime helpers for optional neural inference adapters.

The helpers avoid importing PyTorch at package import time. Memory counters are
best-effort diagnostics: MPS/ CUDA allocator counters do not equal total process
memory and should not be interpreted as complete system-memory usage.
"""
from __future__ import annotations

import gc
from typing import Any


def resolve_torch_device(torch: Any, requested: str = "auto") -> str:
    """Resolve and validate the requested CPU, CUDA or Apple MPS device."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported inference device: {requested!r}")

    cuda_available = bool(
        getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()
    )
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(
        getattr(mps_backend, "is_available", lambda: False)()
    )

    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available in this PyTorch runtime.")
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available in this PyTorch runtime.")
    return requested


def memory_snapshot(torch: Any, device: str | None) -> dict[str, Any]:
    """Collect allocator counters when the active backend exposes them."""
    result: dict[str, Any] = {
        "device": device,
        "available": False,
        "scope": "backend_allocator_counters_not_total_process_memory",
    }
    if device == "mps":
        mps = getattr(torch, "mps", None)
        if mps is None:
            return result
        for name, key in (
            ("current_allocated_memory", "current_allocated_bytes"),
            ("driver_allocated_memory", "driver_allocated_bytes"),
            ("recommended_max_memory", "recommended_max_bytes"),
        ):
            method = getattr(mps, name, None)
            if callable(method):
                try:
                    result[key] = int(method())
                except Exception:
                    result[key] = None
        result["available"] = any(
            result.get(key) is not None
            for key in (
                "current_allocated_bytes",
                "driver_allocated_bytes",
                "recommended_max_bytes",
            )
        )
        return result

    if device == "cuda":
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return result
        for name, key in (
            ("memory_allocated", "current_allocated_bytes"),
            ("memory_reserved", "current_reserved_bytes"),
            ("max_memory_allocated", "peak_allocated_bytes"),
            ("max_memory_reserved", "peak_reserved_bytes"),
        ):
            method = getattr(cuda, name, None)
            if callable(method):
                try:
                    result[key] = int(method())
                except Exception:
                    result[key] = None
        result["available"] = any(
            result.get(key) is not None
            for key in (
                "current_allocated_bytes",
                "current_reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        )
        return result

    return result


def release_device_cache(torch: Any, device: str | None) -> dict[str, Any]:
    """Collect Python objects and ask the active accelerator allocator to release cache."""
    gc.collect()
    result: dict[str, Any] = {
        "python_gc_run": True,
        "device_cache_cleared": False,
        "device": device,
    }
    if device == "mps":
        mps = getattr(torch, "mps", None)
        empty_cache = getattr(mps, "empty_cache", None)
        if callable(empty_cache):
            try:
                empty_cache()
                result["device_cache_cleared"] = True
            except Exception as exc:
                result["cache_clear_error"] = str(exc)
    elif device == "cuda":
        cuda = getattr(torch, "cuda", None)
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            try:
                empty_cache()
                result["device_cache_cleared"] = True
            except Exception as exc:
                result["cache_clear_error"] = str(exc)
    return result


def is_full_commit_sha(revision: str | None) -> bool:
    """Return True only for a full 40-hex Git commit ID, not a branch or tag."""
    import re

    return isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None


def revision_descriptor(revision: str | None) -> dict[str, Any]:
    """Describe whether model loading uses an immutable commit or a mutable ref."""
    if revision is None or not str(revision).strip():
        return {
            "revision": None,
            "revision_kind": "default_model_ref",
            "immutable_commit_pinned": False,
        }
    value = str(revision).strip()
    return {
        "revision": value,
        "revision_kind": "full_commit_sha" if is_full_commit_sha(value) else "named_ref",
        "immutable_commit_pinned": is_full_commit_sha(value),
    }
