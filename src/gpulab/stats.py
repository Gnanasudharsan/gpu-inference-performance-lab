"""Latency statistics: percentiles, bootstrap confidence intervals, goodput, repeatability.

All inputs are latencies in microseconds. Nothing is ever dropped as an "outlier":
tail behaviour is the point of a latency benchmark.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

PERCENTILES = (50, 90, 95, 99)


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float
    # 95% bootstrap CIs (None when n is too small for the estimate to mean anything)
    p50_ci: tuple[float, float] | None
    p99_ci: tuple[float, float] | None
    lag1_autocorr: float
    #: CIs assume roughly independent samples; flagged False when autocorrelation is high.
    ci_reliable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(samples: Sequence[float] | np.ndarray, q: float) -> float:
    """Linear-interpolated percentile (NumPy's default, the same definition everywhere)."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no samples")
    return float(np.percentile(arr, q))


def lag1_autocorrelation(samples: Sequence[float] | np.ndarray) -> float:
    a = np.asarray(samples, dtype=np.float64)
    if a.size < 3 or float(a.std()) == 0.0:
        return 0.0
    a = a - a.mean()
    return float(np.dot(a[:-1], a[1:]) / np.dot(a, a))


def bootstrap_ci(
    samples: Sequence[float] | np.ndarray,
    q: float,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap CI for the q-th percentile."""
    a = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    stats = np.percentile(a[idx], q, axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize(samples: Sequence[float] | np.ndarray, *, seed: int = 0) -> Summary:
    a = np.asarray(samples, dtype=np.float64)
    if a.size == 0:
        raise ValueError("no samples")
    p = {q: float(np.percentile(a, q)) for q in PERCENTILES}
    r1 = lag1_autocorrelation(a)
    # A p99 CI needs enough mass in the tail: require >= ~10 samples above p99.
    p50_ci = bootstrap_ci(a, 50, seed=seed) if a.size >= 20 else None
    p99_ci = bootstrap_ci(a, 99, seed=seed) if a.size >= 1000 else None
    return Summary(
        n=int(a.size),
        mean=float(a.mean()),
        std=float(a.std(ddof=1)) if a.size > 1 else 0.0,
        min=float(a.min()),
        max=float(a.max()),
        p50=p[50],
        p90=p[90],
        p95=p[95],
        p99=p[99],
        p50_ci=p50_ci,
        p99_ci=p99_ci,
        lag1_autocorr=r1,
        ci_reliable=abs(r1) < 0.2,
    )


def deadline_goodput(
    latencies_us: Sequence[float] | np.ndarray, rows_per_request: int, deadline_us: float
) -> dict[str, float]:
    """Closed-loop goodput: rows from requests that met the deadline, per second of busy time.

    ``throughput`` counts every request; ``goodput`` only counts on-time ones, so a
    configuration that is fast on average but has a fat tail is penalized.
    """
    a = np.asarray(latencies_us, dtype=np.float64)
    if a.size == 0:
        raise ValueError("no samples")
    if deadline_us <= 0:
        raise ValueError("deadline must be positive")
    busy_s = float(a.sum()) * 1e-6
    on_time = int((a <= deadline_us).sum())
    return {
        "deadline_us": float(deadline_us),
        "on_time_fraction": on_time / a.size,
        "miss_rate": 1.0 - on_time / a.size,
        "throughput_rows_per_s": a.size * rows_per_request / busy_s,
        "goodput_rows_per_s": on_time * rows_per_request / busy_s,
    }


def repeatability(trial_medians: Sequence[float]) -> dict[str, float]:
    """Spread of per-trial medians. A CV above a few percent means the run is not trustworthy."""
    m = np.asarray(trial_medians, dtype=np.float64)
    if m.size == 0:
        raise ValueError("no trials")
    mean = float(m.mean())
    cv = float(m.std(ddof=1) / mean) if m.size > 1 and mean > 0 else 0.0
    return {
        "trials": float(m.size),
        "median_of_medians": float(np.median(m)),
        "min_median": float(m.min()),
        "max_median": float(m.max()),
        "cv": cv,
        "max_rel_spread": float((m.max() - m.min()) / mean) if mean > 0 else 0.0,
    }


def speedup(baseline_us: float, candidate_us: float) -> float:
    if baseline_us <= 0 or candidate_us <= 0 or math.isnan(baseline_us + candidate_us):
        raise ValueError("latencies must be positive")
    return baseline_us / candidate_us
