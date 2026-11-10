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


def tuned_launch(rows: int) -> metal.LaunchConfig:
    """Launch config chosen from interleaved tuning on Apple M4 (results/phase3/metal_tuning.json).

    rows <= 4: 8 simdgroups x 2 columns (within 0.4% of the best at rows 1, 2; best at 4).
    rows 5-8:  8 simdgroups x 1 column (best at 8 rows; 4 columns was 27-30% slower there,
               plausibly register pressure from 8x4x2 accumulators per lane).
    """
    return metal.LaunchConfig(nsg=8, cols=2 if rows <= 4 else 1)


class MetalImpl(Implementation):
    name = "metal"
    description = "Fused Metal kernels: norm+gate/up+SiLU GEMV and down+residual GEMV (rows<=8)"

    def __init__(self, cfg: metal.LaunchConfig | None = None) -> None:
        self.cfg = cfg  # None: per-row-count tuned configuration

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
        fixed = self.cfg
        launches = {r: fixed or tuned_launch(r) for r in range(1, metal.MAX_ROWS + 1)}
        for launch in {*launches.values(), metal.DEFAULT_LAUNCH}:
            metal.library(weights.dtype, launch)  # compile during build, not on first request

        def small(h: torch.Tensor) -> torch.Tensor:
            launch = launches[h.shape[0]]
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
