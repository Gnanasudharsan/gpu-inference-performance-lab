import math

import pytest
import torch
import torch.nn.functional as F

from gpulab import impls
from gpulab.correctness import TOLERANCES, compare
from gpulab.impls.eager import rmsnorm
from gpulab.workload import ModelConfig, make_inputs, make_weights, reference_forward

EDGE_SHAPES = [
    # rows, d_model, d_hidden, depth
    (1, 64, 160, 1),  # single row
    (3, 96, 256, 2),  # non power-of-two width
    (17, 200, 520, 3),  # odd rows, odd width
    (64, 512, 1408, 2),  # largest batch of interest
]


def _setup(device, dtype, rows, d, h, depth):
    cfg = ModelConfig(d_model=d, d_hidden=h, depth=depth)
    w = make_weights(cfg, device, dtype)
    x = make_inputs(rows, d, device=device, dtype=dtype)
    return cfg, w, x


@pytest.mark.parametrize("shape", EDGE_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_eager_matches_reference(device, dtype, shape):
    _, w, x = _setup(device, dtype, *shape)
    y = impls.get("eager").build(w)(x)
    assert y.dtype == dtype
    assert y.device.type == device
    rep = compare(y, reference_forward(w, x), dtype)
    assert rep.passed, str(rep)


def test_eager_non_contiguous_and_batched_inputs(device, dtype):
    _, w, _ = _setup(device, dtype, 1, 64, 160, 2)
    wide = make_inputs(6, 128, device=device, dtype=dtype)
    x = wide[:, 1::2]
    assert not x.is_contiguous()
    run = impls.get("eager").build(w)
    rep = compare(run(x), reference_forward(w, x), dtype)
    assert rep.passed, str(rep)
    y3 = run(x.reshape(2, 3, 64))
    assert y3.shape == (2, 3, 64)
    assert torch.equal(y3.reshape(6, 64).cpu(), run(x).cpu())


def test_repeated_execution_is_bitwise_stable_and_pure(device, dtype):
    _, w, x = _setup(device, dtype, 8, 128, 352, 2)
    x0 = x.clone()
    run = impls.get("eager").build(w)
    first = run(x).cpu()
    for _ in range(5):
        assert torch.equal(run(x).cpu(), first)
    assert torch.equal(x.cpu(), x0.cpu()), "runner must not mutate its input"


def test_rejects_wrong_width():
    _, w, _ = _setup("cpu", torch.float32, 2, 64, 160, 1)
    with pytest.raises(ValueError, match="last dim"):
        impls.get("eager").build(w)(torch.zeros(2, 63))


# --- the checker must catch realistic bugs -----------------------------------------------------


def _buggy(w, x, bug):
    cfg = w.config
    h = x
    for norm, wg, wu, wd in (b.tensors() for b in w.blocks):
        n = rmsnorm(h, norm, 1e-1 if bug == "eps" else cfg.eps)
        if bug == "swap":
            wg, wu = wu, wg
        out = F.linear(F.silu(F.linear(n, wg)) * F.linear(n, wu), wd)
        h = out if bug == "no_residual" else h + out
    if bug == "row_shift":
        h = torch.roll(h, 1, dims=0)
    return h


@pytest.mark.parametrize("bug", ["swap", "no_residual", "eps", "row_shift"])
def test_checker_detects_bugs(dtype, bug):
    _, w, x = _setup("cpu", dtype, 4, 64, 160, 2)
    rep = compare(_buggy(w, x, bug), reference_forward(w, x), dtype)
    assert not rep.passed, f"bug {bug!r} not detected: {rep}"


def test_checker_flags_nonfinite_and_shape_mismatch():
    ref = torch.ones(2, 4, dtype=torch.float64)
    bad = torch.ones(2, 4)
    bad[0, 0] = math.nan
    assert not compare(bad, ref, torch.float32).finite
    assert not compare(bad, ref, torch.float32).passed
    with pytest.raises(ValueError, match="shape"):
        compare(torch.ones(4, 2), ref, torch.float32)


def test_tolerances_ordered_by_precision():
    t32, t16, tbf = (TOLERANCES[d] for d in (torch.float32, torch.float16, torch.bfloat16))
    assert t32.rel_l2 < t16.rel_l2 < tbf.rel_l2
    assert t32.max_scaled < t16.max_scaled < tbf.max_scaled
