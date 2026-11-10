# GPU validation plan (NVIDIA)

**Status: not executed.** No NVIDIA GPU is available on the development machine, and no GPU-cloud
spend has been authorized. Nothing in this file is a result.

## Goal

Run the same experiment matrix on one NVIDIA data-center GPU (target: L4, A10G, or H100 if available) to
measure the CUDA-only paths: Triton kernels, CUDA graphs, and `torch.compile(mode="reduce-overhead")`.
Also collect Nsight Systems and Nsight Compute evidence.

## Procedure (single script, bounded spend)

1. Provision one GPU container (Modal or Runpod, CUDA 12.x image, PyTorch from the official wheel index).
2. `pip install -e ".[dev,triton]"`, then `gpulab env --json > results/gpu/env.json`.
3. `pytest -m "cuda or triton"`. The gate is that every correctness test passes before any benchmark runs.
4. `scripts/run_experiments.py --suite gpu` (the same matrix as the local suite plus CUDA-only modes).
5. `nsys profile -t cuda,nvtx,osrt -o results/gpu/nsys/<mode> python -m gpulab.cli bench ...` for eager,
   triton and cuda_graph at batch 1 and batch 32. Commit the summaries (`nsys stats`), not the raw traces.
6. `ncu --set full -k regex:gated|rmsnorm -c 20` on the Triton kernels. Record achieved memory bandwidth
   as a fraction of peak, SM occupancy, and register and shared-memory use.
7. Copy the JSON/Parquet results back and regenerate reports. Tag every record with the GPU name and
   driver from `nvidia-smi`.

## Budget guardrails

- Hard cap agreed with the repository owner before provisioning (proposed: USD 10).
- One GPU, maximum wall-clock 45 minutes, auto-shutdown on completion or failure.
- No persistent endpoints are left running.
