"""Fused Triton kernels: 2 launches per block. NVIDIA GPUs, or CPU via TRITON_INTERPRET=1."""

from __future__ import annotations

import os

import torch

from gpulab.impls.base import Implementation, Runner, flatten_rows
from gpulab.impls.fused import pack_gate_up
from gpulab.workload import ModelWeights


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


class TritonImpl(Implementation):
    name = "triton"
    description = "Fused Triton: norm+gate/up GEMM+SiLU*mul epilogue, down GEMM+residual epilogue"

    def unsupported_reason(self, device: torch.device, dtype: torch.dtype) -> str | None:
        if not triton_available():
            return "triton is not installed (Linux only)"
        if device.type == "cuda":
            return None
        if device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1":
            return None  # correctness only: the interpreter is not a performance target
        return "Triton needs an NVIDIA GPU (or TRITON_INTERPRET=1 on CPU for correctness)"

    def build(self, weights: ModelWeights, rows: int | None = None) -> Runner:
        from gpulab.kernels import triton_kernels as tk

        cfg = weights.config
        packed = pack_gate_up(weights)

        def run(x: torch.Tensor) -> torch.Tensor:
            h = flatten_rows(x, cfg.d_model)  # strided inputs are handled by the kernel
            for norm, w_gu, w_down in packed:
                act = tk.norm_gate_up_silu(h, norm, w_gu, cfg.eps)
                h = tk.down_residual(act, w_down, h)
            return h.reshape(x.shape)

        return run
