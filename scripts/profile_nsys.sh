#!/usr/bin/env bash
# Nsight Systems timeline for one implementation on an NVIDIA host.
# NOT EXECUTED in this repository's development environment (Apple M4, no CUDA).
# Usage: scripts/profile_nsys.sh <impl> <rows> [dtype]
set -euo pipefail
impl=${1:?impl}; rows=${2:?rows}; dtype=${3:-float16}
out=results/gpu/nsys/${impl}_r${rows}_${dtype}
mkdir -p "$(dirname "$out")"
command -v nsys >/dev/null || { echo "nsys not found (install Nsight Systems)"; exit 2; }
# --capture-range=cudaProfilerApi would exclude warm-up; here we keep it simple and filter by
# NVTX range in the stats step.
nsys profile -t cuda,nvtx,osrt,cublas --sample=none --force-overwrite=true -o "$out" \
  .venv/bin/gpulab bench --device cuda --dtype "$dtype" --impls "$impl" --rows "$rows" \
  --iters 200 --trials 1 --no-samples
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
  --output "$out" "$out.nsys-rep"
echo "summaries: ${out}_*.csv (commit these, not the .nsys-rep)"
