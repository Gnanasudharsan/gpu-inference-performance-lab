"""Markdown reports from benchmark JSONL files."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from gpulab.records import ExperimentRecord


def paired_speedup(base_trials: list[float], cand_trials: list[float]) -> dict[str, float]:
    """Speedup from paired per-trial medians (interleaved runs). >1 means candidate is faster."""
    if len(base_trials) != len(cand_trials) or not base_trials:
        raise ValueError("need equal, non-empty trial lists")
    r = np.asarray(base_trials, dtype=np.float64) / np.asarray(cand_trials, dtype=np.float64)
    return {
        "geomean": float(np.exp(np.log(r).mean())),
        "min": float(r.min()),
        "max": float(r.max()),
        "trials": float(r.size),
    }


def _fmt(v: float | None, digits: int = 0) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.{digits}f}"


def benchmark_table(records: Iterable[ExperimentRecord], baseline: str = "eager") -> str:
    recs = [r for r in records if r.kind == "benchmark"]
    groups: dict[str, dict[str, ExperimentRecord]] = defaultdict(dict)
    for r in recs:
        g = (r.params.get("interleaved") or {}).get("group")
        if g:
            groups[g][r.impl] = r
    header = (
        "| device | dtype | impl | rows | mode | host p50 µs | p95 | p99 | device p50 µs "
        "| rows/s (p50) | trial CV | speedup vs "
        + baseline
        + " |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    key = lambda r: (r.device, r.dtype, r.model.get("d_model"), r.rows, r.metrics["mode"], r.impl)  # noqa: E731
    for r in sorted(recs, key=key):
        m = r.metrics
        h, d = m.get("host_us") or {}, m.get("device_us") or {}
        sp = "n/a"
        g = (r.params.get("interleaved") or {}).get("group")
        if g and r.impl != baseline and baseline in groups[g]:
            s = paired_speedup(
                groups[g][baseline].metrics["trial_medians_us"], m["trial_medians_us"]
            )
            sp = f"{s['geomean']:.2f}× [{s['min']:.2f}–{s['max']:.2f}]"
        lines.append(
            f"| {r.device} | {r.dtype} | {r.impl} | {r.rows} | {m['mode']} | {_fmt(h.get('p50'))}"
            f" | {_fmt(h.get('p95'))} | {_fmt(h.get('p99'))} | {_fmt(d.get('p50'))}"
            f" | {_fmt(m.get('throughput_rows_per_s'))} | {m['repeatability']['cv']:.3f} | {sp} |"
        )
    return "\n".join(lines)
