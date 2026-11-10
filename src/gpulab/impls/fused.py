"""Algebraic fusion in plain PyTorch (no custom kernels).

* gate and up weights are packed into one (2H, D) matrix: one GEMM instead of two, and
  the activation is read once;
* RMSNorm uses the native fused ``F.rms_norm`` instead of six elementwise ops;
* the residual add is folded into the down projection with ``addmm`` (beta=1).

This is the first rung of the escalation ladder: what can be won without leaving PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gpulab.impls.base import Implementation, Runner, flatten_rows
from gpulab.workload import ModelWeights


def pack_gate_up(weights: ModelWeights) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per block: (norm weight, packed [W_gate; W_up] of shape (2H, D), W_down)."""
    return [
        (b.norm, torch.cat([b.w_gate, b.w_up], dim=0).contiguous(), b.w_down)
        for b in weights.blocks
    ]


def fused_forward(packed, cfg, h: torch.Tensor) -> torch.Tensor:  # type: ignore[no-untyped-def]
    hidden = cfg.d_hidden
    for norm, w_gu, w_down in packed:
        n = F.rms_norm(h, (cfg.d_model,), norm, cfg.eps)
        gu = F.linear(n, w_gu)
        act = F.silu(gu[:, :hidden]) * gu[:, hidden:]
        h = torch.addmm(h, act, w_down.t())
    return h


class FusedWeightsImpl(Implementation):
    name = "fused_weights"
    description = "PyTorch algebraic fusion: packed gate|up GEMM, native rms_norm, addmm residual"

    def build(self, weights: ModelWeights, rows: int | None = None) -> Runner:
        cfg = weights.config
        packed = pack_gate_up(weights)

        def run(x: torch.Tensor) -> torch.Tensor:
            return fused_forward(packed, cfg, flatten_rows(x, cfg.d_model)).reshape(x.shape)

        return run
