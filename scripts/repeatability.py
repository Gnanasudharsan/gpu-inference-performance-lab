"""Cross-process repeatability: run the same benchmark N times in fresh processes.

Within-run trial CV hides slow drift (clock/thermal state, background load). This script
measures run-to-run spread, which is what a reader comparing two numbers actually faces.

Usage: python scripts/repeatability.py --runs 5 --device mps --dtype float16 --rows 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from gpulab.records import read_jsonl


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--impl", default="eager")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--out", help="write summary JSON here")
    a = p.parse_args()
    medians, p99s = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(a.runs):
            path = Path(tmp) / f"run{i}.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gpulab.cli",
                    "bench",
                    "--device",
                    a.device,
                    "--dtype",
                    a.dtype,
                    "--impls",
                    a.impl,
                    "--rows",
                    str(a.rows),
                    "--iters",
                    str(a.iters),
                    "--trials",
                    "3",
                    "--no-samples",
                    "--save",
                    str(path),
                ],
                check=True,
                capture_output=True,
            )
            (rec,) = read_jsonl(path)
            medians.append(rec.metrics["host_us"]["p50"])
            p99s.append(rec.metrics["host_us"]["p99"])
            device_name = rec.device_name
    m = np.array(medians)
    summary = {
        "impl": a.impl,
        "device": a.device,
        "device_name": device_name,
        "dtype": a.dtype,
        "rows": a.rows,
        "runs": a.runs,
        "p50_per_run_us": medians,
        "p99_per_run_us": p99s,
        "p50_run_cv": float(m.std(ddof=1) / m.mean()),
        "p50_max_over_min": float(m.max() / m.min()),
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
