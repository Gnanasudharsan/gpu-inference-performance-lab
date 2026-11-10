"""Fused Metal kernels for Apple GPUs, compiled at runtime with ``torch.mps.compile_shader``.

Small-batch path (rows <= MAX_ROWS), two dispatches per block:

``norm_gate_up_silu``  RMSNorm + packed gate/up GEMV + SiLU(gate)*up, writing only the
                       (rows, H) activation. RMSNorm is folded in by linearity:
                       (s_r * x_r ⊙ g) · w = s_r * ((x_r ⊙ g) · w), so each threadgroup
                       computes s_r = rsqrt(mean(x_r²)+eps) for its rows (x is a few KB and
                       cache-resident) and scales the dot products in the epilogue. No
                       normalized activations and no gate/up intermediates touch memory.
``down_residual``      down-projection GEMV with the residual add fused into the epilogue.

Each simdgroup (32 lanes) owns COLS output columns. Lanes stride over the reduction
dimension with 4-wide vector loads, so one weight row is read with contiguous, coalesced
128-bit-per-lane accesses, exactly once per forward: this is the bandwidth-optimal
pattern for skinny (memory-bound) matrix products. Partial sums are reduced with
``simd_sum`` (no threadgroup memory, no barriers in the hot loop).

Large-batch path (rows > MAX_ROWS): GEMMs go to MPS (compute starts to matter and Apple's
GEMM uses simdgroup matrix hardware) and only the memory-bound pieces are custom:
``rmsnorm`` (one threadgroup per row) and ``silu_mul`` (elementwise).

Accumulation is always fp32. Vector paths require D % 4 == 0 and H % 4 == 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

MAX_ROWS = 8

_MSL_TYPES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}

_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

#define T {T}
#define T4 {T}4
constant constexpr uint NSG = {NSG};     // simdgroups per threadgroup
constant constexpr uint COLS = {COLS};   // output columns per simdgroup
"""

# v2: one specialization per row count R (compile-time constant, so accumulators are
# statically indexed and live in registers), and the column loop is *inside* the K-loop so
# each x load is reused across COLS columns.
_GEMV = r"""
kernel void norm_gate_up_silu_r{R}(
    device const T* x      [[buffer(0)]],   // (R, D)
    device const T* g      [[buffer(1)]],   // (D)
    device const T* wgu    [[buffer(2)]],   // (2H, D): rows [0,H) gate, [H,2H) up
    device T* out          [[buffer(3)]],   // (R, H)
    constant uint& D       [[buffer(4)]],
    constant uint& H       [[buffer(5)]],
    constant float& eps    [[buffer(6)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint tid   [[thread_index_in_threadgroup]])
{{
    constexpr uint R = {R};
    threadgroup float partial[R][NSG];
    threadgroup float inv_rms[R];
    for (uint r = 0; r < R; ++r) {{
        float ss = 0.0f;
        for (uint k = tid; k < D; k += NSG * 32) {{ float v = float(x[r * D + k]); ss += v * v; }}
        ss = simd_sum(ss);
        if (lane == 0) partial[r][sg] = ss;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < R) {{
        float s = 0.0f;
        for (uint i = 0; i < NSG; ++i) s += partial[tid][i];
        inv_rms[tid] = rsqrt(s / float(D) + eps);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint j0 = (tgid * NSG + sg) * COLS;
    if (j0 >= H) return;
    const uint D4 = D / 4;
    device const T4* x4 = (device const T4*)x;
    device const T4* g4 = (device const T4*)g;
    device const T4* wg[COLS];
    device const T4* wu[COLS];
    for (uint c = 0; c < COLS; ++c) {{
        const uint j = min(j0 + c, H - 1);  // clamp: tail columns compute but never store
        wg[c] = (device const T4*)(wgu + (ulong)j * D);
        wu[c] = (device const T4*)(wgu + (ulong)(H + j) * D);
    }}
    float accg[R][COLS];
    float accu[R][COLS];
    for (uint r = 0; r < R; ++r)
        for (uint c = 0; c < COLS; ++c) {{ accg[r][c] = 0.0f; accu[r][c] = 0.0f; }}
    for (uint k = lane; k < D4; k += 32) {{
        const float4 gk = float4(g4[k]);
        float4 a[COLS];
        float4 b[COLS];
        for (uint c = 0; c < COLS; ++c) {{ a[c] = float4(wg[c][k]); b[c] = float4(wu[c][k]); }}
        for (uint r = 0; r < R; ++r) {{
            const float4 xv = float4(x4[r * D4 + k]) * gk;
            for (uint c = 0; c < COLS; ++c) {{
                accg[r][c] += dot(xv, a[c]);
                accu[r][c] += dot(xv, b[c]);
            }}
        }}
    }}
    for (uint r = 0; r < R; ++r) {{
        for (uint c = 0; c < COLS; ++c) {{
            const float gs = simd_sum(accg[r][c]) * inv_rms[r];
            const float us = simd_sum(accu[r][c]) * inv_rms[r];
            if (lane == 0 && j0 + c < H) out[r * H + j0 + c] = T(gs / (1.0f + exp(-gs)) * us);
        }}
    }}
}}

kernel void down_residual_r{R}(
    device const T* act    [[buffer(0)]],   // (R, H)
    device const T* wd     [[buffer(1)]],   // (D, H)
    device const T* res    [[buffer(2)]],   // (R, D)
    device T* out          [[buffer(3)]],   // (R, D)
    constant uint& D       [[buffer(4)]],
    constant uint& H       [[buffer(5)]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{{
    constexpr uint R = {R};
    const uint i0 = (tgid * NSG + sg) * COLS;
    if (i0 >= D) return;
    const uint H4 = H / 4;
    device const T4* a4 = (device const T4*)act;
    device const T4* w[COLS];
    for (uint c = 0; c < COLS; ++c) w[c] = (device const T4*)(wd + (ulong)min(i0 + c, D - 1) * H);
    float acc[R][COLS];
    for (uint r = 0; r < R; ++r)
        for (uint c = 0; c < COLS; ++c) acc[r][c] = 0.0f;
    for (uint k = lane; k < H4; k += 32) {{
        float4 wk[COLS];
        for (uint c = 0; c < COLS; ++c) wk[c] = float4(w[c][k]);
        for (uint r = 0; r < R; ++r) {{
            const float4 av = float4(a4[r * H4 + k]);
            for (uint c = 0; c < COLS; ++c) acc[r][c] += dot(av, wk[c]);
        }}
    }}
    for (uint r = 0; r < R; ++r) {{
        for (uint c = 0; c < COLS; ++c) {{
            const float s = simd_sum(acc[r][c]);
            const uint i = i0 + c;
            if (lane == 0 && i < D) out[r * D + i] = T(float(res[r * D + i]) + s);
        }}
    }}
}}
"""

_HELPERS = r"""
// ---- large-batch helpers -------------------------------------------------------------

kernel void rmsnorm(
    device const T* x      [[buffer(0)]],
    device const T* g      [[buffer(1)]],
    device T* out          [[buffer(2)]],
    constant uint& D       [[buffer(3)]],
    constant float& eps    [[buffer(4)]],
    uint row   [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint ntid  [[threads_per_threadgroup]])
{{
    threadgroup float partial[32];
    threadgroup float inv;
    device const T* xr = x + (ulong)row * D;
    float ss = 0.0f;
    for (uint k = tid; k < D; k += ntid) {{ float v = float(xr[k]); ss += v * v; }}
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {{
        float s = 0.0f;
        for (uint i = 0; i < (ntid + 31) / 32; ++i) s += partial[i];
        inv = rsqrt(s / float(D) + eps);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = tid; k < D; k += ntid) {{
        out[(ulong)row * D + k] = T(float(xr[k]) * inv * float(g[k]));
    }}
}}

kernel void silu_mul(
    device const T* gu     [[buffer(0)]],   // (R, 2H)
    device T* out          [[buffer(1)]],   // (R, H)
    constant uint& H       [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{{
    const uint r = i / H;
    const uint c = i - r * H;
    const float a = float(gu[(ulong)r * 2 * H + c]);
    const float b = float(gu[(ulong)r * 2 * H + H + c]);
    out[i] = T(a / (1.0f + exp(-a)) * b);
}}
"""


@dataclass(frozen=True)
class LaunchConfig:
    nsg: int = 4  # simdgroups per threadgroup
    cols: int = 2  # output columns per simdgroup

    @property
    def threads(self) -> int:
        return self.nsg * 32

    @property
    def cols_per_tg(self) -> int:
        return self.nsg * self.cols


def source(dtype: torch.dtype, cfg: LaunchConfig) -> str:
    head = _HEADER.format(T=_MSL_TYPES[dtype], NSG=cfg.nsg, COLS=cfg.cols)
    gemvs = "".join(_GEMV.format(R=r) for r in range(1, MAX_ROWS + 1))
    return head + gemvs + _HELPERS


DEFAULT_LAUNCH = LaunchConfig()


@lru_cache(maxsize=32)
def library(dtype: torch.dtype, cfg: LaunchConfig = DEFAULT_LAUNCH) -> Any:
    """Compile (once per dtype/config) and cache the Metal library."""
    if dtype not in _MSL_TYPES:
        raise ValueError(f"no Metal kernel for {dtype}")
    return torch.mps.compile_shader(source(dtype, cfg))


def _grid(n_cols: int, cfg: LaunchConfig) -> int:
    return -(-n_cols // cfg.cols_per_tg)


def norm_gate_up_silu(
    x: torch.Tensor,
    g: torch.Tensor,
    wgu: torch.Tensor,
    eps: float,
    cfg: LaunchConfig,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    rows, d = x.shape
    h = wgu.shape[0] // 2
    if out is None:
        out = torch.empty(rows, h, device=x.device, dtype=x.dtype)
    kernel = getattr(library(x.dtype, cfg), f"norm_gate_up_silu_r{rows}")
    kernel(
        x,
        g,
        wgu,
        out,
        d,
        h,
        eps,
        threads=_grid(h, cfg) * cfg.threads,
        group_size=cfg.threads,
    )
    return out


def down_residual(
    act: torch.Tensor,
    wd: torch.Tensor,
    res: torch.Tensor,
    cfg: LaunchConfig,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    rows, h = act.shape
    d = wd.shape[0]
    if out is None:
        out = torch.empty(rows, d, device=act.device, dtype=act.dtype)
    kernel = getattr(library(act.dtype, cfg), f"down_residual_r{rows}")
    kernel(
        act,
        wd,
        res,
        out,
        d,
        h,
        threads=_grid(d, cfg) * cfg.threads,
        group_size=cfg.threads,
    )
    return out


def rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float) -> torch.Tensor:
    rows, d = x.shape
    out = torch.empty_like(x)
    tpg = 256 if d >= 256 else 32 * max(1, -(-d // 32))
    library(x.dtype).rmsnorm(x, g, out, d, eps, threads=rows * tpg, group_size=tpg)
    return out


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    rows, two_h = gu.shape
    h = two_h // 2
    out = torch.empty(rows, h, device=gu.device, dtype=gu.dtype)
    library(gu.dtype).silu_mul(gu, out, h, threads=rows * h)
    return out
