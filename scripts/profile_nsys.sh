#!/usr/bin/env bash
# Nsight Systems timeline for one implementation on an NVIDIA host.
# First attempted on a Kaggle T4 (2026-11-10): failed because it hard-coded .venv/bin/gpulab.
# Usage: scripts/profile_nsys.sh <impl> <rows> [dtype]
set -euo pipefail
impl=${1:?impl}; rows=${2:?rows}; dtype=${3:-float16}
out=results/gpu/nsys/${impl}_r${rows}_${dtype}
mkdir -p "$(dirname "$out")"
command -v nsys >/dev/null || { echo "nsys not found (install Nsight Systems)"; exit 2; }
# the CLI is in .venv locally but installed into the system Python on hosted runners (Kaggle)
gpulab=$(command -v gpulab || echo .venv/bin/gpulab)
# --capture-range=cudaProfilerApi would exclude warm-up; here we keep it simple and filter by
# NVTX range in the stats step.
nsys profile -t cuda,nvtx,osrt,cublas --sample=none --force-overwrite=true -o "$out" \
  "$gpulab" bench --device cuda --dtype "$dtype" --impls "$impl" --rows "$rows" \
  --iters 200 --trials 1 --no-samples
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
  --output "$out" "$out.nsys-rep"
echo "summaries: ${out}_*.csv (commit these, not the .nsys-rep)"
