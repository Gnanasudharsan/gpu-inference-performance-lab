"""Minimal bf16 probes of the Triton toolchain itself (no gpulab kernels involved).

Used to attribute bf16 failures: if these fail under TRITON_INTERPRET=1, the CPU
interpreter's bf16 emulation is at fault, not the gpulab kernels. On a real GPU they must
pass.
"""

import os

import pytest
import torch

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

pytestmark = pytest.mark.triton
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if INTERPRET:
        return "cpu"
    pytest.skip("needs CUDA or TRITON_INTERPRET=1")


@triton.jit
def _roundtrip(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(x_ptr + offs).to(tl.float32) * 2.0
    tl.store(out_ptr + offs, v.to(out_ptr.dtype.element_ty))


@triton.jit
def _dot(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    a = tl.load(a_ptr + i[:, None] * N + i[None, :])
    b = tl.load(b_ptr + i[:, None] * N + i[None, :])
    tl.store(out_ptr + i[:, None] * N + i[None, :], tl.dot(a, b))


def test_bf16_load_cast_store():
    dev = _device()
    x = torch.linspace(-3, 3, 64).to(dev, torch.bfloat16)
    out = torch.empty_like(x)
    _roundtrip[(1,)](x, out, BLOCK=64)
    torch.testing.assert_close(out.float(), x.float() * 2)


@pytest.mark.xfail(
    INTERPRET and not torch.cuda.is_available(),
    reason="Triton 3.9 CPU interpreter computes bf16 tl.dot incorrectly (measured in CI)",
    strict=True,
)
def test_bf16_dot():
    dev = _device()
    g = torch.Generator().manual_seed(0)
    a = torch.randn(16, 16, generator=g).to(dev, torch.bfloat16)
    b = torch.randn(16, 16, generator=g).to(dev, torch.bfloat16)
    out = torch.empty(16, 16, device=dev, dtype=torch.float32)
    _dot[(1,)](a, b, out, N=16)
    torch.testing.assert_close(out, a.float() @ b.float(), rtol=1e-2, atol=1e-2)
