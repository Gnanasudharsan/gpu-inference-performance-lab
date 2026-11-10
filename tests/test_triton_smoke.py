"""Sanity check that the Triton toolchain (GPU or CPU interpreter) executes a kernel correctly."""

import os

import pytest
import torch

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

pytestmark = pytest.mark.triton


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "cpu"
    pytest.skip("Triton needs a CUDA device or TRITON_INTERPRET=1")


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(
        out_ptr + offs,
        tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
        mask=mask,
    )


@pytest.mark.parametrize("n", [1, 127, 1000])
def test_masked_add(n):
    dev = _device()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, generator=g).to(dev)
    y = torch.randn(n, generator=g).to(dev)
    out = torch.empty_like(x)
    _add_kernel[(triton.cdiv(n, 128),)](x, y, out, n, BLOCK=128)
    torch.testing.assert_close(out, x + y)
