"""Does idle time before a request raise its latency? Interleaved, paired measurement.

Phase 2 suspected a GPU power-state effect but sequential runs were too noisy to confirm it.
Here every trial measures every gap value once, in rotated order, inside one process, so
drift affects all gap values equally. The ratio to the 0-gap condition is paired per trial.

Usage: python scripts/idle_gap_interleaved.py [--impl eager] [--trials 30]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gpulab import impls
from gpulab.device import device_name, parse_dtype, resolve_device, synchronize
from gpulab.provenance import collect_provenance
from gpulab.report import paired_speedup
from gpulab.workload import ModelConfig, make_inputs, make_weights


def spin(us: float) -> None:
    end = time.perf_counter_ns() + int(us * 1e3)
    while time.perf_counter_ns() < end:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--impl", default="eager")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--gaps-us", nargs="+", type=float, default=[0, 500, 2000, 10000])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--per-trial", type=int, default=20)
    p.add_argument("--out", default="results/phase4/idle_gap_interleaved.json")
    a = p.parse_args()
    dev, dt = resolve_device(a.device), parse_dtype(a.dtype)
    w = make_weights(ModelConfig(), dev, dt)
    x = make_inputs(a.rows, 1024, device=dev, dtype=dt)
    run = impls.get(a.impl).build(w, rows=a.rows)
    t_end = time.time() + 1.0
    while time.time() < t_end:
        run(x)
        synchronize(dev)
    medians: dict[float, list[float]] = {g: [] for g in a.gaps_us}
    for t in range(a.trials):
        k = t % len(a.gaps_us)
        for gap in a.gaps_us[k:] + a.gaps_us[:k]:
            s = []
            for _ in range(a.per_trial):
                spin(gap)
                t0 = time.perf_counter_ns()
                run(x)
                synchronize(dev)
                s.append((time.perf_counter_ns() - t0) / 1e3)
            medians[gap].append(float(np.median(s)))
    base = medians[a.gaps_us[0]]
    prov = collect_provenance()
    result = {
        "impl": a.impl,
        "device": dev.type,
        "device_name": device_name(dev),
        "dtype": a.dtype,
        "rows": a.rows,
        "trials": a.trials,
        "per_trial": a.per_trial,
        "median_of_trial_medians_us": {str(g): float(np.median(v)) for g, v in medians.items()},
        # >1 means the gap condition is slower than back-to-back (paired per trial)
        "slowdown_vs_no_gap": {
            str(g): paired_speedup(v, base) for g, v in medians.items() if g != a.gaps_us[0]
        },
        "git_commit": prov.git_commit,
        "git_dirty": prov.git_dirty,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("median_of_trial_medians_us", "slowdown_vs_no_gap")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
