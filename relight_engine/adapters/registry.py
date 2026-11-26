"""Pluggable backend registry so core workflows never hardcode specific model classes."""
from __future__ import annotations

from typing import Any, Callable

BackendFactory = Callable[..., Any]

_NORMAL_BACKENDS: dict[str, BackendFactory] = {}
_INTRINSIC_BACKENDS: dict[str, BackendFactory] = {}
_RELIGHTER_BACKENDS: dict[str, BackendFactory] = {}


def _normalize_name(name: str) -> str:
    cleaned = str(name).strip().lower()
    if not cleaned:
        raise ValueError("Backend name must be non-empty.")
    return cleaned


def register_normal_backend(name: str, factory: BackendFactory) -> None:
    _NORMAL_BACKENDS[_normalize_name(name)] = factory


def register_intrinsic_backend(name: str, factory: BackendFactory) -> None:
    _INTRINSIC_BACKENDS[_normalize_name(name)] = factory


def register_relighter_backend(name: str, factory: BackendFactory) -> None:
    _RELIGHTER_BACKENDS[_normalize_name(name)] = factory


def _create(registry: dict[str, BackendFactory], kind: str, name: str, **kwargs: Any) -> Any:
    key = _normalize_name(name)
    if key not in registry:
        known = ", ".join(sorted(registry)) or "none"
        raise ValueError(f"Unknown {kind} backend {name!r}. Available: {known}")
    return registry[key](**kwargs)


def create_normal_backend(name: str, **kwargs: Any) -> Any:
    return _create(_NORMAL_BACKENDS, "normal", name, **kwargs)


def create_intrinsic_backend(name: str, **kwargs: Any) -> Any:
    return _create(_INTRINSIC_BACKENDS, "intrinsic", name, **kwargs)


def create_relighter_backend(name: str, **kwargs: Any) -> Any:
    return _create(_RELIGHTER_BACKENDS, "relighter", name, **kwargs)


def available_backends() -> dict[str, tuple[str, ...]]:
    return {
        "normals": tuple(sorted(_NORMAL_BACKENDS)),
        "intrinsics": tuple(sorted(_INTRINSIC_BACKENDS)),
        "relighters": tuple(sorted(_RELIGHTER_BACKENDS)),
    }


def _register_builtin_backends() -> None:
    def _marigold_normals(**kwargs: Any) -> Any:
        from .diffusers_normals import DiffusersNormalAdapter
        return DiffusersNormalAdapter(**kwargs)

    def _marigold_intrinsics_lighting(**kwargs: Any) -> Any:
        from .diffusers_intrinsics import DiffusersIntrinsicAdapter
        return DiffusersIntrinsicAdapter(variant_kind="lighting", **kwargs)

    def _marigold_intrinsics_appearance(**kwargs: Any) -> Any:
        from .diffusers_intrinsics import DiffusersIntrinsicAdapter
        return DiffusersIntrinsicAdapter(variant_kind="appearance", **kwargs)

    def _ic_light(**kwargs: Any) -> Any:
        from .ic_light_relighter import ICLightRelighter
        return ICLightRelighter(**kwargs)

    def _ic_light_v2(**kwargs: Any) -> Any:
        from .flux_iclight_relighter import FluxICLightRelighter
        return FluxICLightRelighter(**kwargs)

    register_normal_backend("marigold_normals", _marigold_normals)
    register_intrinsic_backend("marigold_intrinsics_lighting", _marigold_intrinsics_lighting)
    register_intrinsic_backend("marigold_intrinsics_appearance", _marigold_intrinsics_appearance)
    register_relighter_backend("ic_light", _ic_light)
    register_relighter_backend("ic_light_v2", _ic_light_v2)
    register_relighter_backend("flux_ic_light", _ic_light_v2)


_register_builtin_backends()
