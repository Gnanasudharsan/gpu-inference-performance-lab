"""Deterministic synthetic request traces (open-loop arrivals).

A trace is a list of requests with arrival offsets, row counts and deadlines. The same
(kind, rate, duration, seed) always produces the same trace, and traces can be saved and
replayed from JSON Lines, so a serving experiment can be rerun exactly.

Arrival processes:
  ``poisson``   exponential inter-arrival times (independent users)
  ``constant``  fixed inter-arrival time (a periodic feed)
  ``bursty``    two-state on/off modulated Poisson: bursts at ``burst_factor`` x rate
                separated by quiet periods, with the same long-run mean rate
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

KINDS = ("poisson", "constant", "bursty")


@dataclass(frozen=True)
class Request:
    rid: int
    arrival_us: float  # offset from replay start
    rows: int
    deadline_us: float  # relative budget: must complete by arrival + deadline

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class TraceSpec:
    kind: str = "poisson"
    rate_rps: float = 200.0
    duration_s: float = 5.0
    seed: int = 0
    rows_choices: tuple[int, ...] = (1,)
    rows_probs: tuple[float, ...] | None = None
    deadline_us: float = 5000.0
    burst_factor: float = 4.0
    burst_fraction: float = 0.2  # fraction of time spent in bursts (bursty only)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.rate_rps <= 0 or self.duration_s <= 0 or self.deadline_us <= 0:
            raise ValueError("rate, duration and deadline must be positive")
        if any(r < 1 for r in self.rows_choices):
            raise ValueError("rows must be >= 1")
        if self.rows_probs is not None and (
            len(self.rows_probs) != len(self.rows_choices) or abs(sum(self.rows_probs) - 1.0) > 1e-9
        ):
            raise ValueError("rows_probs must match rows_choices and sum to 1")
        if self.kind == "bursty" and not (0 < self.burst_fraction * self.burst_factor < 1 + 1e-9):
            raise ValueError("burst_fraction * burst_factor must be <= 1 for a valid mean rate")


def _arrivals(spec: TraceSpec, rng: np.random.Generator) -> np.ndarray:
    horizon = spec.duration_s * 1e6
    mean_gap = 1e6 / spec.rate_rps
    if spec.kind == "constant":
        return np.arange(0.0, horizon, mean_gap)
    if spec.kind == "poisson":
        n = int(spec.rate_rps * spec.duration_s * 1.5) + 16
        t = np.cumsum(rng.exponential(mean_gap, size=n))
        while t[-1] < horizon:  # extremely unlikely, but never truncate silently
            t = np.concatenate([t, t[-1] + np.cumsum(rng.exponential(mean_gap, size=n))])
        return t[t < horizon]
    # bursty: alternate on/off periods (mean period 100 ms) so the long-run rate is rate_rps
    hi = spec.rate_rps * spec.burst_factor
    lo = spec.rate_rps * (1 - spec.burst_fraction * spec.burst_factor) / (1 - spec.burst_fraction)
    period = 100_000.0
    out: list[np.ndarray] = []
    clock, on = 0.0, False
    while clock < horizon:
        dur = float(
            rng.exponential(period * (spec.burst_fraction if on else 1 - spec.burst_fraction))
        )
        rate = hi if on else lo
        if rate > 0:
            k = rng.poisson(rate * dur / 1e6)
            out.append(np.sort(clock + rng.uniform(0, dur, size=k)))
        clock += dur
        on = not on
    arr = np.concatenate(out) if out else np.empty(0)
    return arr[arr < horizon]


def make_trace(spec: TraceSpec) -> list[Request]:
    rng = np.random.default_rng(spec.seed)
    times = _arrivals(spec, rng)
    rows = rng.choice(np.asarray(spec.rows_choices), size=times.size, p=spec.rows_probs)
    return [
        Request(i, float(t), int(r), spec.deadline_us)
        for i, (t, r) in enumerate(zip(times, rows, strict=True))
    ]


def save_trace(path: str | Path, trace: list[Request]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r.to_dict()) + "\n" for r in trace))


def load_trace(path: str | Path) -> list[Request]:
    return [Request(**json.loads(line)) for line in Path(path).read_text().splitlines() if line]
