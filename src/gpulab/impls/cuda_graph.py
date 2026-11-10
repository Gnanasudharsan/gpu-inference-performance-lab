"""CUDA Graph capture of another implementation, for stable-shape low-latency serving.

The whole forward (all kernel launches) is recorded once and replayed with a single
``cudaGraphLaunch``, removing per-kernel CPU launch overhead. Shapes and pointers are
frozen at capture time, so the runner owns a static input buffer: each call copies the
request into it, replays, and clones the static output (two small device copies, both
included in measurements). A request with a different row count is rejected rather than
silently re-captured.

Lifetime rule: a captured graph keeps reading the exact device addresses it saw at capture
(weights, packed weight copies, static buffers). The runner therefore OWNS everything the graph
references: the inner runner (whose closure holds e.g. Triton's packed gate|up weights), the
weights, and the static buffers. Without this, a caller that drops its own reference lets the
allocator reuse that memory and later replays read garbage. That bug was caught on a real T4:
the first replay was correct, later ones were not (results/gpu/diag-t4-20261110-1514).
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

        return GraphRunner(graph, static_x, static_y, rows, cfg.d_model, keepalive=(inner, weights))


class GraphRunner:
    """Callable that replays a captured graph and owns every tensor the graph reads or writes."""

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_x: torch.Tensor,
        static_y: torch.Tensor,
        rows: int,
        d_model: int,
        keepalive: tuple[object, ...],
    ) -> None:
        self.graph = graph
        self.static_x = static_x
        self.static_y = static_y
        self.rows = rows
        self.d_model = d_model
        self._keepalive = keepalive  # inner runner + weights: must outlive the graph

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = flatten_rows(x, self.d_model)
        if h.shape[0] != self.rows:
            raise ValueError(f"graph captured for {self.rows} rows, got {h.shape[0]}")
        self.static_x.copy_(h)
        self.graph.replay()
        return self.static_y.clone().reshape(x.shape)
