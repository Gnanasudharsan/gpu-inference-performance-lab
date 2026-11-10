# GPU validation plan (NVIDIA)

**Status: ready to run; not executed yet.** It needs the repository owner's Kaggle API token. Nothing
in this file is a result.

## Chosen platform: Kaggle Notebooks, free tier (T4)

| Option | Why it was or was not chosen |
|---|---|
| **Kaggle, T4 (chosen)** | Free (community-reported ~30 GPU-h/week). Available today with no sponsor. Fully scriptable with the Kaggle CLI: push a script kernel, run on a T4 (`--accelerator NvidiaTeslaT4`, a "T4 x2" session), download outputs |
| Northeastern Explorer (H100/H200/A100) | Better hardware, but students cannot request partition access themselves (a sponsor is required). **The upgrade path**: `scripts/gpu_session.py` runs unchanged on a cluster node |
| Google Colab free | Same T4 class, but less predictable (no guaranteed GPU, unpublished limits) and not scriptable |
| Modal / Runpod | Credits or paid; better suited to the live API than to a benchmark run |

### T4 caveats (stated up front)

- **Architecture:** Turing (sm_75), with 320 GB/s nominal bandwidth (NVIDIA datasheet; theoretical).
  Results will not transfer to A100/H100 numerically, only methodologically.
- **No bf16 tensor-core MMA.** The Triton implementation reports bf16 as unsupported on compute
  capability < 8.0 (`experiments/nvidia_t4.toml` excludes it). bf16 Triton stays unverified until an
  Ampere-or-newer run.
- **Shared, virtualized GPU.** Clocks and temperature are recorded per record (`nvidia-smi`), and
  comparisons are interleaved and paired, as on the M4.
- **Nsight Compute will likely fail in a container** (no performance-counter access). Nsight Systems
  runs only if `nsys` exists. Kernel counts and kernel time come from the PyTorch profiler's CUDA
  activity either way.
- **Only one GPU is used** of the two in a T4 x2 session (`cuda:0`).

## What one run does (`scripts/gpu_session.py`)

1. **Provenance:** `gpulab env --json`, `nvidia-smi -q`.
2. **Capability gate:** compute capability must be ≥ 7.5.
3. **Correctness gate:** the full test suite on the GPU, including the CUDA-only tests (no hidden host
   syncs, CUDA-graph shape rejection) and the Triton kernels on real hardware. **If it fails, no
   benchmarks run.**
4. **Experiment:** `experiments/nvidia_t4.toml`.
   - *fp16 at 1–64 rows:* eager, fused_weights, compile, compile_cudagraphs, triton, cuda_graph_eager,
     cuda_graph_triton
   - *fp32 at 1, 8 and 32 rows*
   - *two more model widths*
   - *a serving sweep* with a 2 ms deadline
5. **Kernel profiles:** CUDA kernels and kernel time per forward, per implementation.
6. **Nsight Systems:** eager vs `cuda_graph_triton`, if `nsys` is available.
7. **Archive:** `gpulab_results.tar.gz`, containing `session.json` (each step: ok, failed or skipped,
   with a reason).

## How to run

```bash
# one-time (account owner): phone-verify the Kaggle account; create an API token
# (kaggle.com → Settings → API → Create New Token) and save it as ~/.kaggle/kaggle.json (chmod 600)
scripts/kaggle_run.sh          # pushes the current pushed commit, runs (≤ 3 h cap), downloads,
                               # unpacks into results/gpu/kaggle-t4-<date>/, runs gpulab analyze
scripts/kaggle_run.sh --fetch  # re-download the latest run's outputs only
```

`gpulab analyze` merges `results/gpu/*/gpulab_results/` into `results/summary.json`. The dashboard
then shows an NVIDIA device toggle on the Latency, Serving and Kernel pages, and an NVIDIA results
section on the Overview. With no GPU session ingested, none of that appears.

## On a cluster node (Explorer or similar)

```bash
srun -p gpu --gres=gpu:1 --time=02:00:00 --pty bash        # or an sbatch wrapper
make install && .venv/bin/python scripts/gpu_session.py --config experiments/nvidia.toml \
    --out results/gpu/explorer-$(date +%Y%m%d)/gpulab_results
```

Explorer's `gpu` partition allows 4–8 h jobs, and `gpu-short` (1–2 h) schedules faster. The full
`nvidia.toml` includes bf16 for Ampere or newer.
