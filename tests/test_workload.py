import numpy as np
import pytest
import torch

from gpulab.workload import ModelConfig, describe, make_inputs, make_weights, reference_forward


def _numpy_reference(cfg, weights, x):
    """Independent float64 NumPy implementation used to validate reference_forward."""
    h = x.astype(np.float64)
    for b in weights.blocks:
        g, wg, wu, wd = (t.double().numpy() for t in b.tensors())
        n = h / np.sqrt((h**2).mean(-1, keepdims=True) + cfg.eps) * g
        a, u = n @ wg.T, n @ wu.T
        h = h + ((a / (1 + np.exp(-a))) * u) @ wd.T
    return h


@pytest.mark.parametrize(
    "kwargs", [{"d_model": 0}, {"d_hidden": -1}, {"depth": 0}, {"eps": 0.0}, {"eps": 0.5}]
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError, match="must"):
        ModelConfig(**kwargs)


def test_weights_are_deterministic_and_seed_dependent():
    cfg = ModelConfig(d_model=64, d_hidden=160, depth=2)
    a, b = make_weights(cfg), make_weights(cfg)
    c = make_weights(ModelConfig(d_model=64, d_hidden=160, depth=2, seed=1))
    for ba, bb, bc in zip(a.blocks, b.blocks, c.blocks, strict=True):
        for ta, tb, tc in zip(ba.tensors(), bb.tensors(), bc.tensors(), strict=True):
            assert torch.equal(ta, tb)
            assert not torch.equal(ta, tc)


def test_weights_identical_across_devices(device):
    cfg = ModelConfig(d_model=64, d_hidden=160, depth=2)
    cpu, dev = make_weights(cfg), make_weights(cfg, device)
    for bc, bd in zip(cpu.blocks, dev.blocks, strict=True):
        for tc, td in zip(bc.tensors(), bd.tensors(), strict=True):
            assert td.device.type == device
            assert torch.equal(tc, td.cpu())


def test_inputs_deterministic(device, dtype):
    a = make_inputs(5, 32, seed=7, device=device, dtype=dtype)
    b = make_inputs(5, 32, seed=7, device=device, dtype=dtype)
    assert a.dtype == dtype
    assert torch.equal(a.cpu(), b.cpu())
    with pytest.raises(ValueError, match="rows"):
        make_inputs(0, 32)


@pytest.mark.parametrize(("rows", "d", "h", "depth"), [(1, 16, 40, 1), (5, 48, 96, 3)])
def test_reference_matches_independent_numpy(rows, d, h, depth):
    cfg = ModelConfig(d_model=d, d_hidden=h, depth=depth)
    w = make_weights(cfg)
    x = make_inputs(rows, d)
    ref = reference_forward(w, x)
    assert ref.dtype == torch.float64
    np.testing.assert_allclose(
        ref.numpy(), _numpy_reference(cfg, w, x.numpy()), rtol=1e-12, atol=1e-12
    )


def test_reference_keeps_activations_bounded():
    cfg = ModelConfig(d_model=256, d_hidden=704, depth=16)
    ref = reference_forward(make_weights(cfg), make_inputs(8, 256))
    assert torch.isfinite(ref).all()
    assert 0.3 < float(ref.std()) < 10.0


def test_reference_handles_leading_dims_and_strides():
    cfg = ModelConfig(d_model=32, d_hidden=64, depth=2)
    w = make_weights(cfg)
    x = make_inputs(6, 64)[:, ::2]  # non-contiguous view
    assert not x.is_contiguous()
    flat = reference_forward(w, x)
    lead = reference_forward(w, x.reshape(2, 3, 32))
    assert lead.shape == (2, 3, 32)
    torch.testing.assert_close(lead.reshape(6, 32), flat)


def test_describe_counts():
    cfg = ModelConfig(d_model=1024, d_hidden=2816, depth=4)
    d = describe(cfg, rows=1, dtype="float16")
    assert d["params"] == 4 * (3 * 1024 * 2816 + 1024)
    assert d["weight_bytes"] == 2 * d["params"]
    assert d["flops"] == 4 * 2 * 1024 * 2816 * 3
    # batch-1 inference is deeply memory bound: ~1 FLOP per byte
    assert 0.9 < d["arithmetic_intensity"] < 1.01
    assert describe(cfg, rows=64, dtype="float16")["arithmetic_intensity"] > 50
