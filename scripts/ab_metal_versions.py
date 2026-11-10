"""Interleaved A/B of Metal kernel v1 (commit cb577af) against the current kernels.

v1 is loaded from Git history (``git show cb577af:src/gpulab/kernels/metal.py``), so the
comparison is reproducible without keeping dead code in the package. Both run in the same
process with rotated order (``run_interleaved``), at each one's best launch configuration
from tuning.

Usage: python scripts/ab_metal_versions.py [--rows 1 4 8] [--trials 30]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from gpulab import impls
from gpulab.bench import BenchConfig, run_interleaved
from gpulab.impls.base import Implementation, flatten_rows
from gpulab.impls.fused import pack_gate_up
from gpulab.kernels.metal import LaunchConfig
from gpulab.report import paired_speedup
from gpulab.workload import ModelConfig

V1_COMMIT = "cb577af"


def load_v1():
    src = subprocess.run(
        ["git", "show", f"{V1_COMMIT}:src/gpulab/kernels/metal.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = Path(tempfile.mkdtemp()) / "metal_v1.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("metal_v1", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["metal_v1"] = mod
    spec.loader.exec_module(mod)
    return mod


class V1Impl(Implementation):
    name = "metal_v1"
    description = f"Metal GEMV kernels as of {V1_COMMIT}"

    def __init__(self, mod, cfg):
        self.mod, self.cfg = mod, cfg

    def build(self, weights, rows=None):
        c, m, launch = weights.config, self.mod, self.mod.LaunchConfig(self.cfg.nsg, self.cfg.cols)
        packed = pack_gate_up(weights)

        def run(x):
            h = flatten_rows(x, c.d_model).contiguous()
            for norm, w_gu, w_down in packed:
                act = m.norm_gate_up_silu(h, norm, w_gu, c.eps, launch)
                h = m.down_residual(act, w_down, h, launch)
            return h.reshape(x.shape)

        return run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--out", default="results/phase3/metal_v1_vs_v2.json")
    a = p.parse_args()
    v1_cfg = json.loads(Path("results/phase3/metal_v1_tuning.json").read_text())
    v2_cfg = json.loads(Path("results/phase3/metal_tuning.json").read_text())

    def best(tuning, rows):
        per = tuning["results"][str(rows)]
        name = min(per, key=per.__getitem__)
        nsg, cols = name.removeprefix("metal_n").split("_c")
        return LaunchConfig(int(nsg), int(cols))

    mod = load_v1()
    out = {}
    for rows in a.rows:
        from gpulab.impls.metal_impl import MetalImpl

        v1 = V1Impl(mod, best(v1_cfg, rows))
        v2 = MetalImpl(best(v2_cfg, rows))
        v2.name = "metal_v2"
        for impl in (v1, v2):
            impls._REGISTRY[impl.name] = impl
        cfg = BenchConfig(iters=a.trials, trials=a.trials, burst=50, keep_samples=False)
        recs = run_interleaved(
            ["eager", "metal_v1", "metal_v2"],
            ModelConfig(),
            rows,
            torch.device("mps"),
            torch.float16,
            cfg,
            mode="throughput",
        )
        tm = {r.impl: r.metrics["trial_medians_us"] for r in recs}
        res = {
            "launch_v1": vars(v1.cfg),
            "launch_v2": vars(v2.cfg),
            "p50_us": {r.impl: r.metrics["host_us"]["p50"] for r in recs},
            "v2_over_v1": paired_speedup(tm["metal_v1"], tm["metal_v2"]),
            "v2_over_eager": paired_speedup(tm["eager"], tm["metal_v2"]),
            "v1_over_eager": paired_speedup(tm["eager"], tm["metal_v1"]),
            "git_commit": recs[0].provenance["git_commit"],
            "git_dirty": recs[0].provenance["git_dirty"],
        }
        out[str(rows)] = res
        print(
            rows,
            {k: round(v) for k, v in res["p50_us"].items()},
            "v2/v1 {geomean:.2f}x [{min:.2f}-{max:.2f}]".format(**res["v2_over_v1"]),
        )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
