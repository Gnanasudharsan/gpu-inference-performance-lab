"""Measure the eager baseline's error vs. the float64 reference to calibrate tolerances.

Usage: python scripts/calibrate_tolerances.py [--devices cpu mps]
"""

from __future__ import annotations

import argparse
import itertools
import json

import torch

from gpulab import impls
from gpulab.correctness import Tolerance, compare
from gpulab.device import available_devices
from gpulab.workload import ModelConfig, make_inputs, make_weights, reference_forward

SHAPES = [(256, 768, 1), (512, 1408, 4), (1024, 2816, 4), (2048, 5632, 2), (1024, 2816, 8)]
ROWS = [1, 7, 32, 64]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--devices", nargs="+", default=available_devices())
    args = p.parse_args()
    loose = Tolerance(float("inf"), float("inf"))
    worst: dict[str, dict[str, float]] = {}
    for dev, dt in itertools.product(args.devices, DTYPES):
        key = f"{dev}/{str(dt).removeprefix('torch.')}"
        w_rel = w_max = 0.0
        for (d, h, depth), rows in itertools.product(SHAPES, ROWS):
            cfg = ModelConfig(d_model=d, d_hidden=h, depth=depth)
            w = make_weights(cfg, dev, dt)
            x = make_inputs(rows, d, device=dev, dtype=dt)
            y = impls.get("eager").build(w)(x)
            rep = compare(y, reference_forward(w, x), dt, loose)
            w_rel, w_max = max(w_rel, rep.rel_l2), max(w_max, rep.max_scaled)
        worst[key] = {"rel_l2": w_rel, "max_scaled": w_max}
        print(f"{key:22s} worst rel_l2={w_rel:.2e} worst max_scaled={w_max:.2e}", flush=True)
    print(json.dumps(worst))


if __name__ == "__main__":
    main()
