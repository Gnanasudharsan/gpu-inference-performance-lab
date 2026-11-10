"""Custom Metal kernels on Apple GPUs (MPS backend).

rows <= MAX_ROWS : 2 dispatches per block (fused norm+gate/up+SiLU GEMV, fused down+residual GEMV)
rows >  MAX_ROWS : custom rmsnorm -> MPS GEMM (packed gate|up) -> custom silu_mul -> addmm residual
"""

from __future__ import annotations

import torch

from gpulab.impls.base import Implementation, Runner, flatten_rows
from gpulab.impls.fused import pack_gate_up
from gpulab.kernels import metal
from gpulab.workload import ModelWeights


class MetalImpl(Implementation):
    name = "metal"
    description = "Fused Metal kernels: norm+gate/up+SiLU GEMV and down+residual GEMV (rows<=8)"

    def __init__(self, cfg: metal.LaunchConfig | None = None) -> None:
        self.cfg = cfg or metal.LaunchConfig()

    def unsupported_reason(self, device: torch.device, dtype: torch.dtype) -> str | None:
        if device.type != "mps":
            return "Metal kernels run only on Apple GPUs (MPS)"
        if not hasattr(torch.mps, "compile_shader"):
            return "this PyTorch build lacks torch.mps.compile_shader"
        return None

    def build(self, weights: ModelWeights, rows: int | None = None) -> Runner:
        cfg = weights.config
        if cfg.d_model % 4 or cfg.d_hidden % 4:
            raise ValueError("metal kernels need d_model and d_hidden divisible by 4")
        packed = pack_gate_up(weights)
        launch = self.cfg
        metal.library(weights.dtype, launch)  # compile once, during build
        metal.library(weights.dtype)

        def small(h: torch.Tensor) -> torch.Tensor:
            for norm, w_gu, w_down in packed:
                act = metal.norm_gate_up_silu(h, norm, w_gu, cfg.eps, launch)
                h = metal.down_residual(act, w_down, h, launch)
            return h

        def large(h: torch.Tensor) -> torch.Tensor:
            for norm, w_gu, w_down in packed:
                n = metal.rmsnorm(h, norm, cfg.eps)
                act = metal.silu_mul(torch.mm(n, w_gu.t()))
                h = torch.addmm(h, act, w_down.t())
            return h

        def run(x: torch.Tensor) -> torch.Tensor:
            h = flatten_rows(x, cfg.d_model).contiguous()
            out = small(h) if h.shape[0] <= metal.MAX_ROWS else large(h)
            return out.reshape(x.shape)

        return run
