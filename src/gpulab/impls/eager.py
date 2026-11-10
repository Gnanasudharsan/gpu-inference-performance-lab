"""Baseline: idiomatic, unfused eager PyTorch.

Per block this issues separate kernels for the RMSNorm (upcast, square, mean,
rsqrt, scale, downcast), the gate and up GEMMs, SiLU, the gate product, the down
GEMM and the residual add. This is the "correct first" starting point.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gpulab.impls.base import Implementation, Runner, flatten_rows
from gpulab.workload import ModelWeights


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * inv * weight.float()).to(x.dtype)


class EagerImpl(Implementation):
    name = "eager"
    description = "Unfused eager PyTorch (reference baseline)"

    def build(self, weights: ModelWeights, rows: int | None = None) -> Runner:
        cfg = weights.config
        blocks = [b.tensors() for b in weights.blocks]

        def run(x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            h = flatten_rows(x, cfg.d_model)
            for norm, w_gate, w_up, w_down in blocks:
                n = rmsnorm(h, norm, cfg.eps)
                h = h + F.linear(F.silu(F.linear(n, w_gate)) * F.linear(n, w_up), w_down)
            return h.reshape(shape)

        return run
