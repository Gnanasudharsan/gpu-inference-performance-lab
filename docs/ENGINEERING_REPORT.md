# Engineering report

**Question.** How far can a correct PyTorch baseline for small-batch, latency-sensitive inference be
pushed on the GPU that is actually available? And which parts of the system decide the answer?

**Scope of evidence.** Everything *measured* here comes from one Apple M4 laptop (CPU and 10-core GPU
via Metal/MPS, PyTorch 2.15.0). NVIDIA-specific work (Triton kernels, CUDA graphs, Nsight) is
implemented and correctness-tested, but **unmeasured**. The generated, exhaustive tables are in
[`results/ANALYSIS.md`](../results/ANALYSIS.md). This page is the narrative.

## 1. Bottlenecks, as measured before optimizing

| Observation | Evidence |
|---|---|
| Eager issues 64 launches per forward; 12 are GEMMs | `results/phase2/profile_eager_mps.json` |
| At 1–8 rows the work is memory-bound: arithmetic intensity is ~1–8 FLOP/byte | `gpulab.workload.describe` (derived) |
| Eager streams weights at 55–66% of nominal DRAM bandwidth at 1–8 rows | Phase 3 roofline (derived from measured) |
| Synchronous latency tails are clustered (lag-1 autocorrelation up to 0.8); run-to-run p50 varies up to 2.2× | `results/phase2/repeatability_mps.json` |
| End-to-end service is ~2 ms against ~0.8 ms of compute (H2D, sync, D2H) | Phase 2 e2e mode, Phase 4 calibration |

The first two rows predicted where kernel work would pay off: fewer launches and fewer intermediate
round trips at small batch. The last three predicted what kernel work could *not* fix.

## 2. What worked

| Change | Result (pipelined, fp16, paired 95% CI) |
|---|---|
| Fused Metal GEMV kernels (norm folded in by linearity; SiLU·mul and residual epilogues) | **1.16–1.26× at 1–8 rows**; launches 64 → 8; bandwidth 55% → 68% of nominal |
| Same kernels, small model (512×1408) | **1.66× / 1.62×** at 1 / 8 rows |
| Same kernels, fp32 / bf16 at 8 rows | 1.36× / 1.31× |
| v2 kernel rewrite (row-count specialization, column-blocked K-loop) | **1.67× over v1 at 8 rows** (paired geomean; per-trial range 1.29–2.09) |
| Dynamic batching (serving) | 4–5× the peak goodput of direct execution; lowest miss rate under bursts for every implementation |

**Reproducibility.** A separate run through the config-driven runner, about an hour later from a
different commit, replicated the headline speedups: 1.153× vs 1.162× at 1 row and 1.210× vs 1.205× at
8 rows, with overlapping CIs (`results/phase5/repro_comparison.json`).

## 3. What did not work, or stopped working

| Attempt | Outcome | Why (evidence or labeled inference) |
|---|---|---|
| Metal kernel v1 at 8 rows | **0.77× vs eager** (regression) | The column loop sat outside the K-loop, so `x` was re-read per column (~4× more loads than weights at R=8). Fixed in v2 |
| Metal GEMV beyond 8 rows | not attempted; falls back to MPS GEMM | At 16 rows the fallback is **no faster** than eager (1.00×). The GEMV formulation is wrong once compute matters |
| `torch.compile` on CPU | **0.63–0.78×** (regression) | Inductor's CPU code for skinny GEMMs loses to the vendor BLAS used by eager (inference) |
| `torch.compile` on MPS | mostly n.s.; 1.08–1.14× at 32–64 rows | GEMMs dominate and stay vendor GEMMs; fused elementwise work is a small share |
| `fused_weights` on CPU at 8 rows | 0.79× (latency mode) | packed (2H×D) GEMM slower on this CPU (inference, not profiled further) |
| Static batching (serving) at < 300 rps | misses **every** deadline | filling 8 rows takes 8/rate ≫ 10 ms |
| Dynamic batching "wait until the last moment" | missed deadlines in tests | wake-up jitter; fixed with a p90 prediction plus slack margin |
| Display-frame hypothesis for MPS stalls | **rejected** | stall durations are not quantized to 8.3 ms |
| Kernel speedups transferring to serving | mostly did not | serving is bound by the synchronous host↔GPU round trip, not by compute |

## 4. Hardware constraints and what they imply

- **Unified memory, a shared display GPU, aggressive power management.** About 7% of synchronous
  round trips stall for 3–10 ms, and rare batches stall for 114–392 ms. With a 10 ms deadline, a 1%
  miss budget is reachable only at ~100 rps. On this platform the operating envelope is set by the
  environment more than by the kernels.
- **The idle-gap penalty is real.** A 2 ms gap before a request made it 1.19× and 1.35× slower in two
  paired runs. Sparse real-time traffic pays a premium, so batching windows or keep-warm strategies
  are worth their power cost.
- **No device-side MPS timeline.** Launch counts come from instrumented launchers and ATen leaf
  dispatches, and device time from MPS events. There is no Nsight-equivalent kernel-level view on
  this machine.

## 5. Observed vs. theoretical vs. unavailable

| Category | Items |
|---|---|
| **Observed (measured)** | latency distributions, pipelined time, device-event time, e2e stage breakdown, memory, build/compile time, launch counts, serving outcomes, correctness errors |
| **Derived** | speedups and CIs, achieved bandwidth, goodput, miss rates, envelopes |
| **Theoretical** | 120 GB/s nominal M4 bandwidth and the resulting floor (0.58 ms for 69 MB) |
| **Unavailable** | every NVIDIA number (Triton kernel time, CUDA-graph launch savings, `reduce-overhead`), Nsight Systems/Compute traces, PTX/SASS inspection, multi-GPU |

## 6. What I would do next (in order of expected value)

1. **Run [`experiments/nvidia.toml`](../experiments/nvidia.toml)** on one data-center GPU within an
   agreed budget ([GPU_VALIDATION_PLAN.md](GPU_VALIDATION_PLAN.md)). Collect Nsight Systems
   timelines for eager vs CUDA-graph replay, and Nsight Compute DRAM throughput for the Triton
   kernels at M ≤ 16.
2. **Pipeline the serving round trip**: double-buffered H2D/compute/D2H, or device-resident inputs.
   The measurements say this is the largest lever on this platform.
3. **Reduce weight bytes** (int8 or fp8 weights with fp32 accumulation). The forward is
   bandwidth-bound, so this is the largest kernel-level lever, but it needs its own accuracy study.
4. Split-K for the narrow down projection in Triton; compute the norm statistic once per forward.
5. Repeated serving runs per sweep point with CIs; a deadline-derived queue bound; a stall watchdog.
