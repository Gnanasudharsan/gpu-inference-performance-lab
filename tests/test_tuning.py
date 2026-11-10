"""Triton-free tuning helpers and profiler event handling (no GPU needed)."""

from types import SimpleNamespace

import pytest
import torch

from gpulab.kernels.tuning import autotune_m_key
from gpulab.profiling import split_cuda_events
from gpulab.serving.server import BUCKETS, bucket_for


@pytest.mark.parametrize(("m", "key"), [(1, 1), (2, 2), (3, 4), (5, 8), (8, 8), (9, 16), (64, 64)])
def test_autotune_key_is_next_power_of_two(m, key):
    assert autotune_m_key(m) == key


def test_autotune_keys_are_covered_by_serving_calibration_buckets():
    """Calibration warms every bucket, so no row count can autotune mid-serving."""
    for m in range(1, BUCKETS[-1] + 1):
        assert autotune_m_key(m) == bucket_for(m)


def test_split_cuda_events_excludes_record_function_spans():
    cuda, cpu = torch.autograd.DeviceType.CUDA, torch.autograd.DeviceType.CPU

    def ev(name, dev, ann=False, us=1.0):
        return SimpleNamespace(
            name=name, device_type=dev, is_user_annotation=ann, device_time_total=us
        )

    events = [
        ev("gpulab.forward", cuda, ann=True, us=900.0),  # GPU-side copy of the range
        ev("gpulab.forward", cpu),
        ev("ampere_sgemm", cuda, us=10.0),
        ev("vectorized_elementwise_kernel", cuda, us=2.0),
        ev("gpulab.other", cuda, ann=False, us=50.0),  # older torch without the flag
    ]
    kernels, spans = split_cuda_events(events)
    assert [k.name for k in kernels] == ["ampere_sgemm", "vectorized_elementwise_kernel"]
    assert len(spans) == 2
