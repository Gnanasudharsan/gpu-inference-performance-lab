"""Triton kernel correctness. Runs on CUDA, or on CPU with TRITON_INTERPRET=1 (CI)."""

import os

import pytest
import torch

pytest.importorskip("triton")

from gpulab import impls
from gpulab.correctness import compare
from gpulab.workload import ModelConfig, make_inputs, make_weights, reference_forward

pytestmark = pytest.mark.triton


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch.device("cpu")
    pytest.skip("Triton needs a CUDA device or TRITON_INTERPRET=1")


DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).removeprefix("torch."))
@pytest.mark.parametrize(
    ("rows", "d", "h", "depth"),
    [(1, 64, 160, 1), (3, 96, 256, 2), (17, 200, 520, 1), (40, 128, 352, 2)],
    ids=lambda v: str(v),
)
def test_triton_matches_reference(dtype, rows, d, h, depth):
    dev = _device()
    impl = impls.get("triton")
    if not impl.supports(dev, dtype):
        pytest.skip(impl.unsupported_reason(dev, dtype))
    w = make_weights(ModelConfig(d_model=d, d_hidden=h, depth=depth), dev, dtype)
    x = make_inputs(rows, d, device=dev, dtype=dtype)
    y = impl.build(w)(x)
    rep = compare(y, reference_forward(w, x), dtype)
    assert rep.passed, str(rep)


def test_triton_strided_input():
    dev = _device()
    w = make_weights(ModelConfig(96, 256, 1), dev, torch.float32)
    x = make_inputs(5, 192, device=dev)[:, 1::2]
    assert not x.is_contiguous()
    y = impls.get("triton").build(w)(x)
    assert compare(y, reference_forward(w, x), torch.float32).passed


def test_kernels_individually():
    from gpulab.kernels import triton_kernels as tk

    dev = _device()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(7, 72, generator=g).to(dev)
    nw = (1 + 0.1 * torch.randn(72, generator=g)).to(dev)
    wgu = (torch.randn(2 * 40, 72, generator=g) / 8).to(dev)
    act = tk.norm_gate_up_silu(x, nw, wgu, 1e-6)
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * nw
    gate, up = (n @ wgu.T).chunk(2, dim=-1)
    torch.testing.assert_close(act, torch.nn.functional.silu(gate) * up, rtol=1e-4, atol=1e-5)
    wd = (torch.randn(72, 40, generator=g) / 6).to(dev)
    out = tk.down_residual(act, wd, x)
    torch.testing.assert_close(out, x + act @ wd.T, rtol=1e-4, atol=1e-5)
