#!/usr/bin/env bash
# Phase 3 measurement matrix on the local machine (Apple M4: MPS + CPU).
# Every comparison is interleaved (rotated order per trial) so speedups are paired.
set -euo pipefail
G=${G:-.venv/bin/gpulab}
OUT=${OUT:-results/phase3}
mkdir -p "$OUT"
rm -f "$OUT"/bench_*.jsonl
GPU_IMPLS="eager fused_weights compile metal"
# fp16 across batch sizes (default 1024x2816x4 model)
$G bench --device mps --dtype float16 --interleave --impls $GPU_IMPLS \
  --rows 1 2 4 8 16 32 64 --modes latency throughput --iters 400 --trials 20 \
  --deadline-us 2000 --save "$OUT/bench_mps_fp16.jsonl" > /dev/null
# other precisions
for dt in float32 bfloat16; do
  $G bench --device mps --dtype $dt --interleave --impls $GPU_IMPLS \
    --rows 1 8 32 --modes latency throughput --iters 400 --trials 20 \
    --save "$OUT/bench_mps_${dt}.jsonl" > /dev/null
done
# other model widths
$G bench --device mps --dtype float16 --interleave --impls $GPU_IMPLS --rows 1 8 \
  --d-model 512 --d-hidden 1408 --modes latency throughput --iters 400 --trials 20 \
  --save "$OUT/bench_mps_fp16_d512.jsonl" > /dev/null
$G bench --device mps --dtype float16 --interleave --impls $GPU_IMPLS --rows 1 8 \
  --d-model 2048 --d-hidden 5632 --modes latency throughput --iters 200 --trials 20 \
  --save "$OUT/bench_mps_fp16_d2048.jsonl" > /dev/null
# CPU
$G bench --device cpu --dtype float32 --interleave --impls eager fused_weights compile \
  --rows 1 8 32 --modes latency throughput --iters 100 --trials 10 --burst 10 \
  --save "$OUT/bench_cpu_fp32.jsonl" > /dev/null
# dispatch counts per implementation
for impl in $GPU_IMPLS; do
  for rows in 8 32; do
    $G profile --device mps --dtype float16 --impl $impl --rows $rows \
      --save "$OUT/profile_${impl}_r${rows}_mps.json" > /dev/null 2>&1
  done
done
$G report "$OUT"/bench_*.jsonl --out "$OUT/REPORT.md" --parquet "$OUT/phase3.parquet"
echo "done: $OUT/REPORT.md"
