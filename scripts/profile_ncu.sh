#!/usr/bin/env bash
# Nsight Compute kernel analysis for the Triton kernels on an NVIDIA host.
# NOT EXECUTED in this repository's development environment (Apple M4, no CUDA).
# Usage: scripts/profile_ncu.sh <rows> [dtype]
set -euo pipefail
rows=${1:?rows}; dtype=${2:-float16}
out=results/gpu/ncu/triton_r${rows}_${dtype}
mkdir -p "$(dirname "$out")"
command -v ncu >/dev/null || { echo "ncu not found (install Nsight Compute)"; exit 2; }
ncu --set full -k "regex:(rmsnorm|gate_up|down_proj)" -c 30 --force-overwrite -o "$out" \
  .venv/bin/gpulab run --device cuda --dtype "$dtype" --impl triton --rows "$rows"
ncu --import "$out.ncu-rep" --csv --page details \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis > "$out.csv"
echo "summary: $out.csv"
