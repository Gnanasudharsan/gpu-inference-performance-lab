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

VARIANTS = {
    "A_current_eager": "current CudaGraphImpl('eager')",
    "B_current_triton": "current CudaGraphImpl('triton') (control: no cuBLAS)",
    "C_eager_capture_on_warmup_stream": "eager; warm up and capture on the same explicit stream",
    "D_eager_cublas_not_lt": "eager; torch.backends.cuda.preferred_blas_library('cublas')",
    "E_eager_make_graphed_callables": "eager via torch.cuda.make_graphed_callables",
    "F_eager_workspace_env": "current eager with CUBLAS_WORKSPACE_CONFIG=:4096:8",
    "G_eager_cublaslt_workspace_0": "current eager with CUBLASLT_WORKSPACE_SIZE=0",
}


def run_variant(name: str, strided: bool) -> dict:
    import torch

    from gpulab import impls
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

    if name.startswith("D_"):
        torch.backends.cuda.preferred_blas_library("cublas")

    if name.startswith(("A_", "D_", "F_", "G_")):
        run = CudaGraphImpl("eager").build(w, rows=rows)
    elif name.startswith("B_"):
        run = CudaGraphImpl("triton").build(w, rows=rows)
    elif name.startswith("C_"):
        inner = impls.get("eager").build(w)
        static_x = torch.zeros(rows, d, device=dev, dtype=dt)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                inner(static_x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            static_y = inner(static_x)

        def run(xx):
            static_x.copy_(xx.reshape(rows, d))
            g.replay()
            return static_y.clone()
    elif name.startswith("E_"):
        inner = impls.get("eager").build(w)
        sample = torch.zeros(rows, d, device=dev, dtype=dt)
        graphed = torch.cuda.make_graphed_callables(inner, (sample,))

        def run(xx):
            return graphed(xx.reshape(rows, d).contiguous()).clone()
    else:
        raise SystemExit(f"unknown variant {name}")

    y1 = run(x)
    torch.cuda.synchronize()
    kept = y1.clone().cpu()
    sentinels = [
        torch.full((n,), 7.0, device=dev, dtype=torch.float32)
        for n in (256, 4096, 65536, 1 << 20, 1 << 22)
        for _ in range(4)
    ]
    torch.cuda.synchronize()
    outs = []
    for _ in range(5):
        outs.append(run(x).cpu())
    torch.cuda.synchronize()
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
            if name.startswith("F_"):
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            if name.startswith("G_"):
                env["CUBLASLT_WORKSPACE_SIZE"] = "0"
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
