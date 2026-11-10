"""Every optimized implementation must match the float64 reference wherever it is supported."""

import pytest
import torch

from gpulab import impls
from gpulab.correctness import compare
from gpulab.workload import ModelConfig, make_inputs, make_weights, reference_forward

SHAPES = [
    # rows, d_model, d_hidden, depth
    (1, 64, 160, 1),
    (3, 96, 256, 2),
    (8, 200, 520, 2),  # largest row count on the Metal GEMV path
    (17, 128, 352, 2),  # Metal large-batch path, odd rows
    (64, 256, 704, 1),
]
# torch.compile recompiles per shape; keep its coverage small to bound test time.
COMPILE_SHAPES = [SHAPES[1], SHAPES[3]]
OPTIMIZED = [n for n in impls.names() if n != "eager"]


def _check(name, device, dtype, rows, d, h, depth, strided=False):
    impl = impls.get(name)
    dev = torch.device(device)
    if not impl.supports(dev, dtype):
        pytest.skip(impl.unsupported_reason(dev, dtype))
    w = make_weights(ModelConfig(d_model=d, d_hidden=h, depth=depth), dev, dtype)
    if strided:
        x = make_inputs(rows, 2 * d, device=dev, dtype=dtype)[:, ::2]
        assert not x.is_contiguous()
    else:
        x = make_inputs(rows, d, device=dev, dtype=dtype)
    run = impl.build(w, rows=rows)
    y = run(x)
    assert y.shape == x.shape
    assert y.dtype == dtype
    rep = compare(y, reference_forward(w, x), dtype)
    assert rep.passed, f"{name} {device} {dtype} {rows}x{d}x{h}x{depth}: {rep}"
    return run, x, y


@pytest.mark.parametrize("name", OPTIMIZED)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_matches_reference(name, device, dtype, shape):
    if name.startswith("compile") and shape not in COMPILE_SHAPES:
        pytest.skip("compile coverage limited to two shapes")
    _check(name, device, dtype, *shape)


@pytest.mark.parametrize("name", OPTIMIZED)
def test_strided_input_and_repeatability(name, device, dtype):
    run, x, y = _check(name, device, dtype, 5, 96, 256, 2, strided=True)
    for _ in range(3):
        assert torch.equal(run(x).cpu(), y.cpu()), "repeated execution must be bitwise stable"


@pytest.mark.parametrize("name", [n for n in OPTIMIZED if not n.startswith("compile")])
def test_leading_dims(name, device):
    impl = impls.get(name)
    dev, dt = torch.device(device), torch.float32
    if not impl.supports(dev, dt) or impl.static_shape:
        pytest.skip("unsupported or static-shape")
    w = make_weights(ModelConfig(64, 160, 1), dev, dt)
    x = make_inputs(6, 64, device=dev, dtype=dt)
    run = impl.build(w)
    torch.testing.assert_close(run(x.reshape(2, 3, 64)).reshape(6, 64), run(x))


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
@pytest.mark.parametrize(("nsg", "cols"), [(1, 1), (2, 4), (4, 2), (8, 1), (8, 4)])
@pytest.mark.parametrize("rows", [1, 5, 8])
def test_metal_launch_configs(nsg, cols, rows):
    from gpulab.impls.metal_impl import MetalImpl
    from gpulab.kernels.metal import LaunchConfig

    dev, dt = torch.device("mps"), torch.float16
    w = make_weights(ModelConfig(200, 520, 2), dev, dt)
    x = make_inputs(rows, 200, device=dev, dtype=dt)
    y = MetalImpl(LaunchConfig(nsg=nsg, cols=cols)).build(w)(x)
    rep = compare(y, reference_forward(w, x), dt)
    assert rep.passed, str(rep)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_metal_rejects_unaligned_widths():
    w = make_weights(ModelConfig(50, 128, 1), "mps", torch.float16)
    with pytest.raises(ValueError, match="divisible by 4"):
        impls.get("metal").build(w)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA GPU")
def test_cuda_graph_rejects_other_shapes():
    w = make_weights(ModelConfig(64, 160, 1), "cuda", torch.float16)
    run = impls.get("cuda_graph_eager").build(w, rows=4)
    with pytest.raises(ValueError, match="captured for 4 rows"):
        run(make_inputs(5, 64, device="cuda", dtype=torch.float16))


def test_every_impl_reports_support_consistently():
    for name in impls.names():
        impl = impls.get(name)
        for dev in ("cpu", "mps", "cuda"):
            reason = impl.unsupported_reason(torch.device(dev), torch.float16)
            assert reason is None or isinstance(reason, str)
