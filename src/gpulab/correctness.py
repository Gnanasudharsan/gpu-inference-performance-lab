"""Correctness checking against the float64 reference.

Errors are measured relative to the reference's scale so that thresholds do not
depend on the absolute magnitude of activations:

* ``rel_l2``      = ||y - ref||_2 / ||ref||_2   (aggregate accuracy)
* ``max_scaled``  = max|y - ref| / max|ref|     (worst single element)

Thresholds were calibrated empirically (see docs/CORRECTNESS.md). They are a few times
looser than the eager baseline's own error in each dtype and tight enough to catch
real bugs such as a missing residual, a wrong mask, or a transposed weight, which all
produce errors of O(1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from gpulab.device import dtype_name, parse_dtype


@dataclass(frozen=True)
class Tolerance:
    rel_l2: float
    max_scaled: float


TOLERANCES: dict[torch.dtype, Tolerance] = {
    torch.float32: Tolerance(rel_l2=1e-5, max_scaled=1e-4),
    torch.float16: Tolerance(rel_l2=3e-3, max_scaled=1e-2),
    torch.bfloat16: Tolerance(rel_l2=2e-2, max_scaled=8e-2),
}


@dataclass(frozen=True)
class ErrorReport:
    dtype: str
    rel_l2: float
    max_scaled: float
    max_abs: float
    finite: bool
    tolerance: Tolerance
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict} [{self.dtype}] rel_l2={self.rel_l2:.2e} (tol {self.tolerance.rel_l2:.0e})"
            f" max_scaled={self.max_scaled:.2e} (tol {self.tolerance.max_scaled:.0e})"
        )


def compare(
    out: torch.Tensor,
    ref: torch.Tensor,
    dtype: torch.dtype | str,
    tolerance: Tolerance | None = None,
) -> ErrorReport:
    dt = parse_dtype(dtype)
    tol = tolerance or TOLERANCES[dt]
    if out.shape != ref.shape:
        raise ValueError(f"shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}")
    y = out.detach().cpu().to(torch.float64)
    r = ref.detach().cpu().to(torch.float64)
    finite = bool(torch.isfinite(y).all())
    diff = (y - r).abs()
    ref_norm = float(r.norm()) or 1.0
    ref_max = float(r.abs().max()) or 1.0
    rel_l2 = float((y - r).norm()) / ref_norm if finite else float("inf")
    max_abs = float(diff.max()) if finite else float("inf")
    max_scaled = max_abs / ref_max
    passed = finite and rel_l2 <= tol.rel_l2 and max_scaled <= tol.max_scaled
    return ErrorReport(dtype_name(dt), rel_l2, max_scaled, max_abs, finite, tol, passed)
