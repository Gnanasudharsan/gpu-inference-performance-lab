"""Implementation interface shared by every execution mode."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from gpulab.workload import ModelWeights

Runner = Callable[[torch.Tensor], torch.Tensor]


class Implementation(ABC):
    """An execution strategy for the gated-MLP stack.

    ``build`` does all one-time work (weight packing, compilation, graph capture,
    shader compilation) so that the returned runner only does per-request work.
    Runners accept inputs of shape ``(..., d_model)`` with arbitrary strides on the
    weights' device and dtype, and return a tensor of the same shape.
    """

    name: str = ""
    description: str = ""
    #: True if ``build`` specialises on the input row count (e.g. CUDA graphs).
    static_shape: bool = False

    def unsupported_reason(self, device: torch.device, dtype: torch.dtype) -> str | None:
        """Return None if supported, otherwise a human-readable reason."""
        return None

    def supports(self, device: torch.device, dtype: torch.dtype) -> bool:
        return self.unsupported_reason(device, dtype) is None

    @abstractmethod
    def build(self, weights: ModelWeights, rows: int | None = None) -> Runner:
        """Prepare a runner. ``rows`` is required when ``static_shape`` is True."""


def flatten_rows(x: torch.Tensor, d_model: int) -> torch.Tensor:
    if x.shape[-1] != d_model:
        raise ValueError(f"expected last dim {d_model}, got shape {tuple(x.shape)}")
    return x.reshape(-1, d_model)
