"""CUDA Graph capture of another implementation, for stable-shape low-latency serving.

The whole forward (all kernel launches) is recorded once and replayed with a single
``cudaGraphLaunch``, removing per-kernel CPU launch overhead. Shapes and pointers are
frozen at capture time, so the runner owns a static input buffer: each call copies the
request into it, replays, and clones the static output (two small device copies, both
included in measurements). A request with a different row count is rejected rather than
silently re-captured.
"""

from __future__ import annotations

import torch

from gpulab import impls as registry
from gpulab.impls.base import Implementation, Runner, flatten_rows
from gpulab.workload import ModelWeights


class CudaGraphImpl(Implementation):
    static_shape = True

    def __init__(self, inner: str) -> None:
        self.inner = inner
        self.name = f"cuda_graph_{inner}"
        self.description = f"CUDA Graph replay of the '{inner}' implementation"

    def unsupported_reason(self, device: torch.device, dtype: torch.dtype) -> str | None:
        if device.type != "cuda":
            return "CUDA graphs require an NVIDIA GPU"
        return registry.get(self.inner).unsupported_reason(device, dtype)

    def build(self, weights: ModelWeights, rows: int | None = None) -> Runner:
        if rows is None:
            raise ValueError(f"{self.name} needs a fixed row count at build time")
        cfg = weights.config
        inner = registry.get(self.inner).build(weights, rows)
        static_x = torch.zeros(rows, cfg.d_model, device=weights.device, dtype=weights.dtype)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # warm up: autotuning, lazy init and allocator state before capture
                inner(static_x)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_y = inner(static_x)

        def run(x: torch.Tensor) -> torch.Tensor:
            h = flatten_rows(x, cfg.d_model)
            if h.shape[0] != rows:
                raise ValueError(f"graph captured for {rows} rows, got {h.shape[0]}")
            static_x.copy_(h)
            graph.replay()
            return static_y.clone().reshape(x.shape)

        return run
