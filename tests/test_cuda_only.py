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
