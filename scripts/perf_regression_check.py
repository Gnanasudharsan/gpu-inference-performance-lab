"""GPU performance regression gate (local GPU only).

Re-measures eager vs metal (pipelined, fp16, rows 1 and 8) through the experiment runner and
fails if a speedup falls below the recorded Phase 3 CI lower bound by more than a margin,
or loses its "faster" verdict.

Usage: python scripts/perf_regression_check.py [--margin 0.05]
"""

from __future__ import annotations

import argparse
import glob
import json
import tempfile
from pathlib import Path

from gpulab.analysis import comparisons
from gpulab.experiment import load_experiment, run_experiment
from gpulab.records import read_jsonl


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--out", default="results/phase8/perf_regression.json")
    a = p.parse_args()
    base = {
        c.rows: c
        for c in comparisons(
            [r for f in glob.glob("results/phase3/bench_mps_fp16.jsonl") for r in read_jsonl(f)]
        )
        if c.mode == "throughput" and c.candidate == "metal" and c.rows in (1, 8)
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_experiment(load_experiment("experiments/repro_check.toml"), tmp, log=lambda *_: None)
        new = {c.rows: c for c in comparisons(read_jsonl(Path(tmp) / "bench.jsonl"))}
    rows = {}
    ok = True
    for r, b in sorted(base.items()):
        n = new[r]
        floor = b.ci_low * (1 - a.margin)
        passed = n.speedup_geomean >= floor and n.verdict == "faster"
        ok &= passed
        rows[r] = {
            "recorded": [b.speedup_geomean, b.ci_low, b.ci_high],
            "now": [n.speedup_geomean, n.ci_low, n.ci_high],
            "floor": floor,
            "verdict_now": n.verdict,
            "passed": passed,
        }
    out = {"margin": a.margin, "rows": rows, "passed": ok}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
