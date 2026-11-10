"""Checks that only make sense on NVIDIA hardware. Skipped elsewhere (no GPU in CI)."""

import pytest
import torch

from gpulab import impls
from gpulab.workload import ModelConfig, make_inputs, make_weights

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an NVIDIA GPU"),
]


@pytest.mark.parametrize("name", impls.names())
def test_runners_have_no_hidden_host_syncs(name):
    """A hidden device->host sync (e.g. .item(), .cpu(), a data-dependent branch) in the hot
    path serializes the host with the GPU and destroys launch pipelining."""
    impl = impls.get(name)
    dev, dt = torch.device("cuda"), torch.float16
    if not impl.supports(dev, dt):
        pytest.skip(impl.unsupported_reason(dev, dt))
    w = make_weights(ModelConfig(256, 704, 2), dev, dt)
    x = make_inputs(8, 256, device=dev, dtype=dt)
    run = impl.build(w, rows=8)
    run(x)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run(x)
    finally:
        torch.cuda.set_sync_debug_mode("default")


GRAPH_IMPLS = [n for n in impls.names() if n.startswith("cuda_graph") or n == "compile_cudagraphs"]


@pytest.mark.parametrize("name", GRAPH_IMPLS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["float32", "float16"])
def test_graph_runner_owns_what_it_replays(name, dtype):
    """Regression test for the first T4 session (results/gpu/diag-t4-20261110-1514).

    A captured graph keeps reading the addresses it saw at capture. Here every outside
    reference to the weights is dropped and freed memory is overwritten with NaN before
    replaying, so a runner that does not own its inputs fails deterministically, not by luck.
    """
    import gc

    from gpulab.correctness import compare
    from gpulab.workload import reference_forward

    impl = impls.get(name)
    dev = torch.device("cuda")
    if not impl.supports(dev, dtype):
        pytest.skip(impl.unsupported_reason(dev, dtype))
    w = make_weights(ModelConfig(96, 256, 2), dev, dtype)
    x = make_inputs(10, 192, device=dev, dtype=dtype)[:, ::2]  # strided, as in the failing case
    ref = reference_forward(w, x)
    run = impl.build(w, rows=10)
    first = run(x)
    torch.cuda.synchronize()
    kept = first.clone().cpu()
    del w
    gc.collect()
    scribble = [
        torch.full((n,), float("nan"), device=dev, dtype=dtype)
        for n in (96 * 256, 2 * 96 * 256, 256 * 96, 4096, 65536)
        for _ in range(8)
    ]
    later = [run(x) for _ in range(5)]
    torch.cuda.synchronize()
    del scribble
    assert torch.equal(first.cpu(), kept), "a replay modified an output already returned"
    for y in later:
        rep = compare(y, ref, dtype)
        assert rep.passed, f"replay after dropping outside references: {rep}"
