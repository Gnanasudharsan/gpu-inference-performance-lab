"""Registry of execution modes."""

from __future__ import annotations

from gpulab.impls.base import Implementation, Runner
from gpulab.impls.compiled import CompileCudaGraphsImpl, CompileImpl
from gpulab.impls.cuda_graph import CudaGraphImpl
from gpulab.impls.eager import EagerImpl
from gpulab.impls.fused import FusedWeightsImpl
from gpulab.impls.metal_impl import MetalImpl
from gpulab.impls.triton_impl import TritonImpl

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


for _impl in (
    EagerImpl(),
    FusedWeightsImpl(),
    CompileImpl(),
    MetalImpl(),
    TritonImpl(),
    CudaGraphImpl("eager"),
    CudaGraphImpl("triton"),
    CompileCudaGraphsImpl(),
):
    register(_impl)

__all__ = ["Implementation", "Runner", "get", "names", "register"]
