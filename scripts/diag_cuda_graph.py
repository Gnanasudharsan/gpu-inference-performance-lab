"""Diagnose CUDA-graph replay memory safety (NVIDIA only).

Background: on a Kaggle T4 (torch 2.11+cu130) the cuda_graph_eager implementation passed a
single-call correctness check, but an output it had already returned changed after later
replays, while cuda_graph_triton (no cuBLAS) was stable. Hypothesis: the captured graph writes
cuBLAS workspace into memory the caching allocator considers free, so replays can corrupt other
tensors.

Each variant runs in a fresh subprocess (cuBLAS workspace state is process-global) and reports:
  kept_output_changed  an output returned by an earlier replay was modified by later replays
  sentinels_corrupted  tensors allocated after the first replay were modified by later replays
  outputs_correct      every replay output matches the float64 reference within tolerance
  outputs_stable       all replay outputs are bitwise identical

Usage: python scripts/diag_cuda_graph.py --out results/gpu/diag
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# v1 (commit 6345adb) compared capture/cuBLAS variants and disproved the workspace hypothesis:
# the eager graph was clean in isolation while the Triton graph broke, which pointed to graph
# LIFETIMES (the runner did not own what the graph reads). v2 checks the fixed runner, with the
# caller's references kept or dropped (dropped = weights deleted, freed memory scribbled with NaN).
VARIANTS = {
    "eager_refs_kept": "fixed CudaGraphImpl('eager'), caller keeps the weights",
    "eager_refs_dropped": "fixed CudaGraphImpl('eager'), caller drops the weights",
    "triton_refs_kept": "fixed CudaGraphImpl('triton'), caller keeps the weights",
    "triton_refs_dropped": "fixed CudaGraphImpl('triton'), caller drops the weights",
}


def run_variant(name: str, strided: bool) -> dict:
    import gc

    import torch

    from gpulab.correctness import compare
    from gpulab.impls.cuda_graph import CudaGraphImpl
    from gpulab.workload import ModelConfig, make_inputs, make_weights, reference_forward

    dev, dt, rows, d = torch.device("cuda"), torch.float16, 5, 96
    cfg = ModelConfig(d_model=d, d_hidden=256, depth=2)
    w = make_weights(cfg, dev, dt)
    if strided:
        x = make_inputs(rows, 2 * d, device=dev, dtype=dt)[:, ::2]
    else:
        x = make_inputs(rows, d, device=dev, dtype=dt)
    ref = reference_forward(w, x)
    run = CudaGraphImpl(name.split("_")[0]).build(w, rows=rows)

    y1 = run(x)
    torch.cuda.synchronize()
    kept = y1.clone().cpu()
    scribble = []
    if name.endswith("dropped"):
        del w
        gc.collect()
        scribble = [
            torch.full((n,), float("nan"), device=dev, dtype=dt)
            for n in (d * 256, 2 * d * 256, 4096, 65536)
            for _ in range(8)
        ]
    sentinels = [
        torch.full((n,), 7.0, device=dev, dtype=torch.float32)
        for n in (256, 4096, 65536, 1 << 20, 1 << 22)
        for _ in range(4)
    ]
    torch.cuda.synchronize()
    outs = [run(x).cpu() for _ in range(5)]
    torch.cuda.synchronize()
    del scribble
    return {
        "variant": name,
        "strided_input": strided,
        "kept_output_changed": not torch.equal(y1.cpu(), kept),
        "sentinels_corrupted": sum(int(not bool((t == 7.0).all())) for t in sentinels),
        "outputs_correct": all(compare(o, ref, dt).passed for o in [kept, *outs]),
        "outputs_stable": all(torch.equal(o, outs[0]) for o in outs),
        "first_output_correct": compare(kept, ref, dt).passed,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/gpu/diag")
    p.add_argument("--variant")
    p.add_argument("--strided", action="store_true")
    a = p.parse_args()
    if a.variant:  # child process
        print(json.dumps(run_variant(a.variant, a.strided)))
        return 0
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env_info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    results = []
    for name, desc in VARIANTS.items():
        for strided in (True, False):
            env = dict(os.environ)
            cmd = [sys.executable, __file__, "--variant", name] + (["--strided"] if strided else [])
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
            try:
                rec = json.loads(r.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                rec = {"variant": name, "strided_input": strided, "error": r.stderr[-1500:]}
            rec["description"] = desc
            results.append(rec)
            print(json.dumps(rec), flush=True)
    (out / "diag_cuda_graph.json").write_text(
        json.dumps({"env": env_info, "results": results}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
