# Benchmark methodology

The harness lives in [`src/gpulab/bench.py`](../src/gpulab/bench.py) and the statistics in
[`src/gpulab/stats.py`](../src/gpulab/stats.py). This page explains what each number means, how it is
obtained, and which biases it is and is not protected against.

## What is measured

| Metric | Definition | How |
|---|---|---|
| **host latency** (`host_us`) | Wall-clock time a caller waits for one synchronous request: Python dispatch + kernel launches + device execution + synchronization | `perf_counter_ns` around `runner(x); synchronize()` |
| **device time** (`device_us`) | Span between a device event recorded before and after the request on the device timeline | `torch.cuda.Event` / `torch.mps.Event`, measured in a **separate pass** (see below) |
| **end-to-end latency** (`e2e` mode) | Host tensor in, host tensor out: H2D copy + compute + D2H copy + sync | host timer; per-stage breakdown from an instrumented pass |
| **pipelined time per request** (`throughput` mode) | Mean time per request when `burst` requests are issued back to back with one sync | host timer per burst ÷ burst size |
| **throughput** | rows ÷ p50 host latency (rows/s) for synchronous serving | derived |
| **goodput** | Rows from requests that **met the deadline**, per second of busy time | `stats.deadline_goodput` |
| **build time / first call** | One-time cost: weight packing, `torch.compile`, shader compilation, graph capture; the cold first call includes lazy initialization (e.g. Metal pipeline creation) | reported separately, never mixed into steady-state latency |
| **memory** | Allocator bytes before and after; driver-allocated bytes on MPS; peak allocated on CUDA | `torch.{cuda,mps}` allocator APIs |

The difference between host latency and device time is **host-side overhead**: Python, dispatcher,
launches and synchronization. The difference between latency mode and throughput mode shows how much
of that overhead pipelining can hide. At small batch this gap is often the whole story.

## Distribution statistics

- p50, p90, p95, p99, min, max, mean and std from all samples. Samples are **never** discarded as outliers.
- 95% percentile-bootstrap confidence intervals. The p50 CI needs n ≥ 20. The p99 CI needs n ≥ 1000,
  so that about 10 samples lie beyond p99.
- **Lag-1 autocorrelation** of the sample series. Bootstrap CIs assume independent samples. When
  |r₁| ≥ 0.2, the record sets `ci_reliable = false` and the CI must not be quoted as a precise bound.
- Raw samples are stored in each record (`samples_host_us`) so distributions can be re-analyzed.

## Controls against measurement bias

| Bias | Control |
|---|---|
| Asynchronous execution (timing only the launch) | Synchronize before stopping the host timer. Device time comes from events |
| Cold caches, lazy init, compilation | Warm-up by count (≥ 30) **and** time (≥ 1 s). Build and first-call time recorded separately |
| Clock ramp-up (DVFS) | Time-based warm-up under sustained load. 0.25 s was measured to be insufficient on Apple M4 |
| Instrumentation perturbation | On MPS, recording events every request was measured to raise host latency and its variance. Host latency is measured **without** events, and device time in a separate pass |
| Python GC pauses | `gc.collect()` then `gc.disable()` around timed loops |
| Slow drift (thermal, background load, power state) | Multiple trials. For comparisons, `--interleave` rotates the implementation order every trial and speedups are computed from **paired** trial medians |
| Run-to-run variation | `scripts/repeatability.py` repeats a benchmark in fresh processes |
| Silent wrong answers | Every benchmark first checks correctness against the float64 reference and refuses to time a failing implementation |
| Environment conditions | Each record stores power source, low-power mode, thermal and performance warnings (macOS `pmset`), or SM/memory clocks, temperature and power (`nvidia-smi`) before and after |

## Measured caveats on the development machine (Apple M4 laptop)

These were observed on 2026-11-10 and drove the controls above.

1. **The latency tail is clustered, not random.** In 2,000 back-to-back synchronous requests (eager,
   fp16, 8 rows), lag-1 autocorrelation was 0.71. Medians of 100-request blocks alternated between
   about 1.22 ms and about 3.48 ms. The GPU moves between fast and slow states over hundreds of
   milliseconds. This laptop's GPU also composites a 3024×1964 display for other applications, so p99
   on this machine reflects the environment, not only the kernels.
2. **Idle gaps before a request raise its latency** (GPU power-state behavior). See
   `results/phase2/idle_gap_mps.jsonl`. Sparse-arrival latency is therefore measured explicitly with
   `--idle-gap-us`, not inferred from back-to-back runs.
3. **Within-run trials agree better than separate runs.** One run gave a trial-median CV below 1%,
   while separate processes differed by about 25% in p50. Cross-process spread is measured in
   `results/phase2/repeatability_mps.json`.
4. **Unified memory.** H2D and D2H on Apple silicon are copies within shared DRAM plus command
   scheduling. They do not model PCIe transfers on discrete GPUs.

## NVIDIA-specific considerations (for the GPU validation run)

- Lock clocks if permitted (`nvidia-smi -lgc`), or at least record them. Boost clocks vary with
  temperature and power.
- Use `torch.cuda.set_sync_debug_mode("error")` (see `tests/test_cuda_only.py`) to show the hot path
  has no hidden device-to-host synchronizations.
- Disable TF32 for fp32 comparisons. Triton kernels use `input_precision="ieee"` for fp32.
- Report `torch.compile` and Triton autotuning time as build time.
- Use Nsight Systems to attribute host-latency minus device time to launches (CUDA API rows), and
  Nsight Compute to compare achieved DRAM bandwidth with peak.

## Sample size

The default is 300 requests per configuration in 3 trials. Comparisons use `--interleave` with 10–20
short trials. p99 from 300 samples rests on only 3 samples beyond it and is reported without a CI.
Any p99 claim in this repository uses ≥ 1,000 samples or is labeled indicative.

## Reproduce

```bash
gpulab bench --device mps --dtype float16 --rows 1 8 32 --modes latency e2e throughput \
  --iters 600 --trials 10 --deadline-us 2000 --save results/phase2/baseline_mps.jsonl
gpulab bench --device mps --dtype float16 --rows 8 --idle-gap-us 2000 --save idle.jsonl
gpulab bench --interleave --impls eager <candidate> --rows 1 8 32 --trials 20 --iters 600
python scripts/repeatability.py --runs 5 --device mps --dtype float16 --rows 8
gpulab report results/phase2/*.jsonl --out results/phase2/REPORT.md --parquet results/phase2/phase2.parquet
```
