"""Fused Triton kernels for NVIDIA GPUs (also runnable on CPU with TRITON_INTERPRET=1).

``norm_gate_up_silu``  out = SiLU(s ⊙ (x⊙g) W_gateᵀ) * (s ⊙ (x⊙g) W_upᵀ), s = rsqrt(mean(x²)+eps)
    One program computes a BLOCK_M × BLOCK_N tile of the activation. The K-loop streams
    x⊙g tiles through ``tl.dot`` against the matching gate *and* up weight tiles (two fp32
    accumulators, one read of x), and accumulates Σx² per row in the same pass, so RMSNorm
    costs no extra memory traffic and no extra launch. The (rows, 2H) gate/up intermediate
    never exists in memory.
``down_residual``      out = res + act W_downᵀ  (residual add in the epilogue)

Precision: tile inputs are in the working dtype (Tensor Core eligible), accumulation is
fp32, and fp32 inputs use ``input_precision="ieee"`` so TF32 is never used silently.

Known limitation: at very small M the grid is only ceil(N / BLOCK_N) programs. A split-K
or stream-K variant would raise SM occupancy for the narrow down projection, but it
needs a two-phase reduction because the epilogues are nonlinear (see docs/KERNELS.md).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from gpulab.kernels.tuning import autotune_m_key

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

if INTERPRET:
    # The interpreter executes tiles with NumPy; one small config keeps CI fast.
    _CONFIGS = [triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32})]
else:
    _CONFIGS = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
        for bm in (16, 32, 64)
        for bn in (32, 64, 128)
        for bk in (64, 128)
        for w, s in ((4, 3), (4, 4), (8, 3))
        if not (bm == 64 and bn == 128 and bk == 128)
    ]


@triton.autotune(configs=_CONFIGS, key=["M_KEY", "D", "H"])
@triton.jit
def _norm_gate_up_silu_kernel(
    x_ptr,
    g_ptr,
    w_ptr,
    out_ptr,
    M,
    D,
    H,
    M_KEY,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    eps,
    IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < H
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
        gk = tl.load(g_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        xg = (x * gk[None, :]).to(w_ptr.dtype.element_ty)
        w_off = offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = mask_k[:, None] & mask_n[None, :]
        wg = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)
        wu = tl.load(w_ptr + H * stride_wn + w_off, mask=w_mask, other=0.0)
        if IEEE:
            acc_g = tl.dot(xg, wg, acc_g, input_precision="ieee")
            acc_u = tl.dot(xg, wu, acc_u, input_precision="ieee")
        else:
            acc_g = tl.dot(xg, wg, acc_g)
            acc_u = tl.dot(xg, wu, acc_u)
    inv = 1.0 / tl.sqrt(ss / D + eps)
    gs = acc_g * inv[:, None]
    us = acc_u * inv[:, None]
    y = gs * tl.sigmoid(gs) * us
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        y.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.autotune(configs=_CONFIGS, key=["M_KEY", "D", "H"])
@triton.jit
def _down_residual_kernel(
    a_ptr,
    w_ptr,
    r_ptr,
    out_ptr,
    M,
    D,
    H,
    M_KEY,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_rm,
    stride_rn,
    stride_om,
    stride_on,
    IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < D
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        if IEEE:
            acc = tl.dot(a, w, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, w, acc)
    mask = mask_m[:, None] & mask_n[None, :]
    res = tl.load(
        r_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn, mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        (acc + res).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def norm_gate_up_silu(
    x: torch.Tensor, g: torch.Tensor, wgu: torch.Tensor, eps: float
) -> torch.Tensor:
    """x: (M, D) any strides; g: (D,); wgu: (2H, D) packed [W_gate; W_up]."""
    M, D = x.shape
    H = wgu.shape[0] // 2
    out = torch.empty(M, H, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(H, meta["BLOCK_N"]))

    _norm_gate_up_silu_kernel[grid](
        x,
        g,
        wgu,
        out,
        M,
        D,
        H,
        autotune_m_key(M),
        x.stride(0),
        x.stride(1),
        wgu.stride(0),
        wgu.stride(1),
        out.stride(0),
        out.stride(1),
        eps,
        IEEE=x.dtype == torch.float32,
    )
    return out


def down_residual(act: torch.Tensor, wd: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """act: (M, H); wd: (D, H); res: (M, D). Returns res + act @ wd.T."""
    M, H = act.shape
    D = wd.shape[0]
    out = torch.empty(M, D, device=act.device, dtype=act.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(D, meta["BLOCK_N"]))

    _down_residual_kernel[grid](
        act,
        wd,
        res,
        out,
        M,
        D,
        H,
        autotune_m_key(M),
        act.stride(0),
        act.stride(1),
        wd.stride(0),
        wd.stride(1),
        res.stride(0),
        res.stride(1),
        out.stride(0),
        out.stride(1),
        IEEE=act.dtype == torch.float32,
    )
    return out
