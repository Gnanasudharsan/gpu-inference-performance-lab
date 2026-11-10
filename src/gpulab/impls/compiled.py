"""torch.compile (Inductor) variants.

``compile``             default mode, static shapes, full graph. On CUDA Inductor emits Triton
                        kernels; on MPS it emits Metal shaders for fused elementwise work.
``compile_cudagraphs``  ``mode="reduce-overhead"``: Inductor plus CUDA graphs (CUDA only).

Compilation time is reported by the harness as build time, never as latency.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import torch

from gpulab.impls.base import Implementation, Runner, flatten_rows
from gpulab.impls.fused import fused_forward, pack_gate_up
from gpulab.workload import ModelWeights


def _fresh_code(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Copy ``fn`` with its own code object.

    Dynamo caches compiled graphs per code object and guards on closure contents, so
    repeated builds of the same inner function would pile up recompiles until the
    recompile limit. Each build gets an independent cache, like a separately deployed model.
    """
    return types.FunctionType(
        fn.__code__.replace(), fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )


class CompileImpl(Implementation):
    name = "compile"
    description = "torch.compile(fullgraph, static) of the fused-weights forward"
    mode: str | None = None
    static_shape = True

    def build(self, weights: ModelWeights, rows: int | None = None) -> Runner:
        cfg = weights.config
        packed = pack_gate_up(weights)

        def fwd(h: torch.Tensor) -> torch.Tensor:
            return fused_forward(packed, cfg, h)

        compiled = torch.compile(_fresh_code(fwd), fullgraph=True, dynamic=False, mode=self.mode)
        dev = weights.device
        if rows is not None:  # compile eagerly so build time includes compilation
            compiled(torch.zeros(rows, cfg.d_model, device=dev, dtype=weights.dtype))

        def run(x: torch.Tensor) -> torch.Tensor:
            return compiled(flatten_rows(x, cfg.d_model).contiguous()).reshape(x.shape)

        return run


class CompileCudaGraphsImpl(CompileImpl):
    name = "compile_cudagraphs"
    description = "torch.compile(mode='reduce-overhead'): Inductor + CUDA graphs"
    mode = "reduce-overhead"

    def unsupported_reason(self, device: torch.device, dtype: torch.dtype) -> str | None:
        return None if device.type == "cuda" else "CUDA graphs require an NVIDIA GPU"

    def build(self, weights: ModelWeights, rows: int | None = None) -> Runner:
        inner = super().build(weights, rows)

        def run(x: torch.Tensor) -> torch.Tensor:
            # outputs of cudagraph-backed modules alias static memory; clone to hand ownership
            # to the caller (one small device copy, included in the measurement)
            return inner(x).clone()

        return run
