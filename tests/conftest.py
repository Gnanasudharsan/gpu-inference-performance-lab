import torch

from gpulab.device import available_devices

DEVICES = available_devices()
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        metafunc.parametrize("device", DEVICES)
    if "dtype" in metafunc.fixturenames:
        metafunc.parametrize("dtype", DTYPES, ids=lambda d: str(d).removeprefix("torch."))
