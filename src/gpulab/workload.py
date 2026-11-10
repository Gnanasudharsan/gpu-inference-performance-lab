"""Workload definition: a stack of pre-norm RMSNorm -> SwiGLU gated MLP -> residual blocks.

Block ``i`` computes, for rows ``x`` of width ``d_model``::

    n = x * rsqrt(mean(x^2) + eps) * g_i                  # RMSNorm
    h = silu(n @ W_gate_i^T) * (n @ W_up_i^T)              # SwiGLU, width d_hidden
    x = x + h @ W_down_i^T                                 # residual

Weights and inputs are generated on the CPU from a fixed seed in float32 and only
then cast and moved. The same seed therefore produces identical values on every
device and dtype, which makes cross-device comparisons meaningful.

Numerical contract for reduced precision (float16/bfloat16), matched by every
implementation in this repository:

* activations and weights are *stored* in the working dtype;
* RMSNorm statistics are computed in float32;
* GEMMs accumulate in float32 (the default for cuBLAS, MPS and Triton ``tl.dot``);
* SiLU and the gate product are evaluated in float32, then rounded to the working dtype.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from gpulab.device import dtype_name, parse_dtype


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 1024
    d_hidden: int = 2816  # ~2.75x, rounded to a multiple of 64 like common SwiGLU sizing
    depth: int = 4
    eps: float = 1e-6
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.d_hidden <= 0 or self.depth <= 0:
            raise ValueError("d_model, d_hidden and depth must be positive")
        if not (0.0 < self.eps < 1e-2):
            raise ValueError("eps must be in (0, 1e-2)")

    @property
    def params_per_block(self) -> int:
        return 3 * self.d_model * self.d_hidden + self.d_model

    @property
    def num_params(self) -> int:
        return self.depth * self.params_per_block

    def flops(self, rows: int) -> int:
        """Matmul FLOPs for one forward pass (2 * M * N * K per GEMM; elementwise ignored)."""
        return self.depth * 2 * rows * self.d_model * self.d_hidden * 3

    def weight_bytes(self, dtype: torch.dtype | str) -> int:
        return self.num_params * parse_dtype(dtype).itemsize

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _move(t: torch.Tensor, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    # Cast on the CPU side of the transfer: MPS has no float64, so casting there fails.
    if torch.device(device).type == "cpu":
        return t.to("cpu").to(dtype).contiguous()
    return t.to(dtype).to(device).contiguous()


@dataclass
class BlockWeights:
    norm: torch.Tensor  # (d_model,)
    w_gate: torch.Tensor  # (d_hidden, d_model)
    w_up: torch.Tensor  # (d_hidden, d_model)
    w_down: torch.Tensor  # (d_model, d_hidden)

    def to(self, device: torch.device | str, dtype: torch.dtype) -> BlockWeights:
        return BlockWeights(*(_move(t, device, dtype) for t in self.tensors()))

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.norm, self.w_gate, self.w_up, self.w_down)


@dataclass
class ModelWeights:
    config: ModelConfig
    blocks: list[BlockWeights] = field(default_factory=list)

    @property
    def device(self) -> torch.device:
        return self.blocks[0].w_gate.device

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].w_gate.dtype

    def to(self, device: torch.device | str, dtype: torch.dtype | str) -> ModelWeights:
        dt = parse_dtype(dtype)
        return ModelWeights(self.config, [b.to(device, dt) for b in self.blocks])


def make_weights(
    config: ModelConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | str = torch.float32,
) -> ModelWeights:
    """Deterministic weights. Scaled so activations stay O(1) through ``depth`` blocks."""
    g = torch.Generator(device="cpu").manual_seed(config.seed)
    d, h = config.d_model, config.d_hidden
    # 1/sqrt(depth) on the down projection keeps the residual stream's variance bounded.
    down_scale = 1.0 / math.sqrt(h) / math.sqrt(config.depth)
    blocks = []
    for _ in range(config.depth):
        blocks.append(
            BlockWeights(
                norm=1.0 + 0.1 * torch.randn(d, generator=g),
                w_gate=torch.randn(h, d, generator=g) / math.sqrt(d),
                w_up=torch.randn(h, d, generator=g) / math.sqrt(d),
                w_down=torch.randn(d, h, generator=g) * down_scale,
            )
        )
    return ModelWeights(config, blocks).to(device, dtype)


def make_inputs(
    rows: int,
    d_model: int,
    seed: int = 1234,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | str = torch.float32,
) -> torch.Tensor:
    """Deterministic (rows, d_model) inputs, generated on CPU then moved."""
    if rows <= 0:
        raise ValueError("rows must be positive")
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, d_model, generator=g)
    return x.to(device=device, dtype=parse_dtype(dtype))


def reference_forward(weights: ModelWeights, x: torch.Tensor) -> torch.Tensor:
    """Golden reference: float64 on CPU, independent of the implementation under test.

    Accepts any shape ``(..., d_model)`` and any strides. The result is float64 on CPU.
    """
    cfg = weights.config
    w64 = weights.to("cpu", torch.float64) if weights.dtype != torch.float64 else weights
    lead = x.shape[:-1]
    h = x.detach().cpu().to(torch.float64).reshape(-1, cfg.d_model)
    for b in w64.blocks:
        n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.eps) * b.norm
        gate = n @ b.w_gate.T
        up = n @ b.w_up.T
        h = h + (torch.nn.functional.silu(gate) * up) @ b.w_down.T
    return h.reshape(*lead, cfg.d_model)


def describe(config: ModelConfig, rows: int, dtype: torch.dtype | str) -> dict[str, Any]:
    """Static roofline-relevant quantities for a forward pass."""
    dt = parse_dtype(dtype)
    wbytes = config.weight_bytes(dt)
    act_bytes = config.depth * rows * (2 * config.d_model + 2 * config.d_hidden) * dt.itemsize
    flops = config.flops(rows)
    return {
        "dtype": dtype_name(dt),
        "rows": rows,
        "params": config.num_params,
        "weight_bytes": wbytes,
        "flops": flops,
        # FLOPs per byte of compulsory DRAM traffic (weights + minimal activation traffic)
        "arithmetic_intensity": flops / (wbytes + act_bytes),
    }
