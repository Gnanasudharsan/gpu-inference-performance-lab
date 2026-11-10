"""Registry of execution modes."""

from __future__ import annotations

from gpulab.impls.base import Implementation, Runner
from gpulab.impls.eager import EagerImpl

_REGISTRY: dict[str, Implementation] = {}


def register(impl: Implementation) -> Implementation:
    if impl.name in _REGISTRY:
        raise ValueError(f"duplicate implementation {impl.name!r}")
    _REGISTRY[impl.name] = impl
    return impl


def get(name: str) -> Implementation:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown implementation {name!r}; have {sorted(_REGISTRY)}") from None


def names() -> list[str]:
    return list(_REGISTRY)


register(EagerImpl())

__all__ = ["Implementation", "Runner", "get", "names", "register"]
