"""Experiment analysis: paired speedups with confidence intervals, regressions, roofline.

Evidence labels are kept explicit:

* ``measured``    numbers read directly from benchmark records;
* ``derived``     arithmetic on measured numbers (speedups, achieved bandwidth);
* ``theoretical`` numbers from specifications (vendor-nominal peak bandwidth), never
                  presented as observations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gpulab.records import ExperimentRecord

#: Vendor-nominal peak DRAM bandwidth (GB/s), used only for *theoretical* floors.
NOMINAL_BANDWIDTH_GBPS: dict[str, float] = {
    "Apple M4": 120.0,  # Apple M4 (base) unified memory, vendor-stated
    "Tesla T4": 320.0,  # NVIDIA T4 datasheet (GDDR6)
}


def nominal_bandwidth(device_name: str) -> float | None:
    for key, bw in NOMINAL_BANDWIDTH_GBPS.items():
        if key in device_name:
            return bw
    return None


@dataclass(frozen=True)
class Comparison:
    device: str
    device_name: str
    dtype: str
    d_model: int
    d_hidden: int
    depth: int
    rows: int
    mode: str
    baseline: str
    candidate: str
    baseline_p50_us: float
    candidate_p50_us: float
    speedup_geomean: float
    ci_low: float
    ci_high: float
    trials: int
    verdict: str  # "faster" | "slower" | "no significant difference"
    candidate_p99_us: float | None
    baseline_p99_us: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_ratio_ci(
    base: list[float], cand: list[float], n_boot: int = 4000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Geometric-mean speedup (base/cand) with a percentile-bootstrap CI over paired trials."""
    if len(base) != len(cand) or len(base) < 2:
        raise ValueError("need >= 2 paired trials")
    logr = np.log(np.asarray(base, dtype=np.float64) / np.asarray(cand, dtype=np.float64))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, logr.size, size=(n_boot, logr.size))
    boots = logr[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.exp(logr.mean())), float(np.exp(lo)), float(np.exp(hi))


def verdict(ci_low: float, ci_high: float) -> str:
    if ci_low > 1.0:
        return "faster"
    if ci_high < 1.0:
        return "slower"
    return "no significant difference"


def comparisons(records: Iterable[ExperimentRecord], baseline: str = "eager") -> list[Comparison]:
    """All paired comparisons against ``baseline`` within interleaved groups."""
    groups: dict[tuple[str, str], dict[str, ExperimentRecord]] = defaultdict(dict)
    for r in records:
        inter = r.params.get("interleaved")
        if r.kind == "benchmark" and inter:
            groups[(inter["group"], r.metrics["mode"])][r.impl] = r
    out = []
    for (_, mode), by_impl in groups.items():
        if baseline not in by_impl:
            continue
        b = by_impl[baseline]
        for name, c in by_impl.items():
            if name == baseline:
                continue
            g, lo, hi = paired_ratio_ci(
                b.metrics["trial_medians_us"], c.metrics["trial_medians_us"]
            )
            m = c.model
            out.append(
                Comparison(
                    device=c.device,
                    device_name=c.device_name,
                    dtype=c.dtype,
                    d_model=m["d_model"],
                    d_hidden=m["d_hidden"],
                    depth=m["depth"],
                    rows=c.rows,
                    mode=mode,
                    baseline=baseline,
                    candidate=name,
                    baseline_p50_us=b.metrics["host_us"]["p50"],
                    candidate_p50_us=c.metrics["host_us"]["p50"],
                    speedup_geomean=g,
                    ci_low=lo,
                    ci_high=hi,
                    trials=len(c.metrics["trial_medians_us"]),
                    verdict=verdict(lo, hi),
                    candidate_p99_us=c.metrics["host_us"]["p99"] if mode == "latency" else None,
                    baseline_p99_us=b.metrics["host_us"]["p99"] if mode == "latency" else None,
                )
            )
    return sorted(out, key=lambda c: (c.device, c.dtype, c.d_model, c.mode, c.candidate, c.rows))


def roofline(record: ExperimentRecord) -> dict[str, Any]:
    """Achieved weight-streaming bandwidth (derived) vs. the nominal floor (theoretical).

    Uses pipelined (throughput-mode) time when available: it excludes per-request sync
    overhead, so it is the right numerator for "how close to the memory limit is the work".
    """
    wl = record.params.get("workload", {})
    p50 = (record.metrics.get("host_us") or {}).get("p50")
    bw_nominal = nominal_bandwidth(record.device_name)
    out: dict[str, Any] = {
        "weight_bytes": wl.get("weight_bytes"),
        "arithmetic_intensity": wl.get("arithmetic_intensity"),
        "achieved_gbps": None,
        "nominal_gbps": bw_nominal,
        "floor_us": None,
        "fraction_of_nominal": None,
    }
    if p50 and wl.get("weight_bytes"):
        achieved = wl["weight_bytes"] / (p50 * 1e-6) / 1e9
        out["achieved_gbps"] = achieved  # derived
        if bw_nominal:
            out["floor_us"] = wl["weight_bytes"] / (bw_nominal * 1e9) * 1e6  # theoretical
            out["fraction_of_nominal"] = achieved / bw_nominal
    return out


def comparison_table(comps: list[Comparison]) -> str:
    lines = [
        "| device | dtype | model | mode | impl | rows | eager p50 µs | impl p50 µs "
        "| speedup [95% CI] | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in comps:
        lines.append(
            f"| {c.device} | {c.dtype} | {c.d_model}×{c.d_hidden}×{c.depth} | {c.mode} "
            f"| {c.candidate} | {c.rows} | {c.baseline_p50_us:,.0f} | {c.candidate_p50_us:,.0f} "
            f"| {c.speedup_geomean:.2f}× [{c.ci_low:.2f}, {c.ci_high:.2f}] | {c.verdict} |"
        )
    return "\n".join(lines)
