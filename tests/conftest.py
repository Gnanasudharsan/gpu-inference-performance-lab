import torch

from gpulab.device import available_devices

DEVICES = available_devices()
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _explicitly_parametrized(metafunc, name):
    return any(
        name in (m.args[0] if isinstance(m.args[0], (list, tuple)) else m.args[0].split(","))
        for m in metafunc.definition.iter_markers("parametrize")
    )


def pytest_generate_tests(metafunc):
    """Default device/dtype parametrization, unless a test parametrizes them itself."""
    if "device" in metafunc.fixturenames and not _explicitly_parametrized(metafunc, "device"):
        metafunc.parametrize("device", DEVICES)
    if "dtype" in metafunc.fixturenames and not _explicitly_parametrized(metafunc, "dtype"):
        metafunc.parametrize("dtype", DTYPES, ids=lambda d: str(d).removeprefix("torch."))
