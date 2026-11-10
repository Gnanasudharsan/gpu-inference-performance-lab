"""Device selection, synchronization and dtype support."""

from __future__ import annotations

import torch

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return DTYPES[name]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r}; choose from {sorted(DTYPES)}") from None


def dtype_name(dtype: torch.dtype) -> str:
    for k, v in DTYPES.items():
        if v == dtype:
            return k
    raise ValueError(f"unsupported dtype {dtype}")


def available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def resolve_device(name: str = "auto") -> torch.device:
    """Map ``auto`` to the best local accelerator; validate explicit names."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(name)
    if dev.type not in available_devices():
        raise RuntimeError(f"device {name!r} is not available here; have {available_devices()}")
    return dev


def synchronize(device: torch.device | str) -> None:
    """Block the host until all queued work on ``device`` has finished."""
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def device_name(device: torch.device | str) -> str:
    from gpulab.provenance import accelerators

    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    for acc in accelerators():
        if acc.backend == dev.type:
            return acc.name
    return dev.type


def memory_allocated(device: torch.device | str) -> int | None:
    """Bytes currently allocated by PyTorch's allocator on ``device`` (None on CPU)."""
    dev = torch.device(device)
    if dev.type == "cuda":
        return int(torch.cuda.memory_allocated(dev))
    if dev.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return None
