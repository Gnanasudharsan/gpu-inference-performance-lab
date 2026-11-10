"""Tune the Metal GEMV launch configuration by interleaved, pipelined measurement.

Every (simdgroups per threadgroup, columns per simdgroup) candidate is registered as a
temporary implementation and measured with ``run_interleaved(mode="throughput")``, so all
candidates see the same clock/thermal conditions. Result: results/phase3/metal_tuning.json.

Usage: python scripts/tune_metal.py [--rows 1 4 8] [--dtype float16] [--trials 20]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from gpulab import impls
from gpulab.bench import BenchConfig, run_interleaved
from gpulab.device import parse_dtype
from gpulab.impls.metal_impl import MetalImpl
from gpulab.kernels.metal import LaunchConfig
from gpulab.workload import ModelConfig

GRID = list(itertools.product((1, 2, 4, 8), (1, 2, 4)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--dtype", default="float16")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--burst", type=int, default=50)
    p.add_argument("--out", default="results/phase3/metal_tuning.json")
    a = p.parse_args()
    names = []
    for nsg, cols in GRID:
        impl = MetalImpl(LaunchConfig(nsg=nsg, cols=cols))
        impl.name = f"metal_n{nsg}_c{cols}"
        impls.register(impl)
        names.append(impl.name)
    cfg = BenchConfig(iters=a.trials, trials=a.trials, burst=a.burst, warmup_s=1.0,
                      keep_samples=False)
    out: dict[str, object] = {"dtype": a.dtype, "model": ModelConfig().to_dict(), "results": {}}
    results: dict[str, dict[str, float]] = {}
    for rows in a.rows:
        recs = run_interleaved(names, ModelConfig(), rows, torch.device("mps"),
                               parse_dtype(a.dtype), cfg, mode="throughput")
        per = {r.impl: r.metrics["host_us"]["p50"] for r in recs}
        results[str(rows)] = per
        best = min(per, key=per.__getitem__)
        print(f"rows={rows}: best {best} {per[best]:.0f} us; worst {max(per.values()):.0f} us")
        for name in sorted(per, key=per.__getitem__)[:5]:
            print(f"    {name:<14} {per[name]:8.1f} us")
    out["results"] = results
    out["device_name"] = recs[0].device_name
    out["git_commit"] = recs[0].provenance["git_commit"]
    out["git_dirty"] = recs[0].provenance["git_dirty"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
