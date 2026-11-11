# NVIDIA T4 results (Kaggle free tier)

The same workload, harness and paired method as on the Apple M4, run on an NVIDIA Tesla T4 through
`scripts/kaggle_run.sh`. Figures below come from the session that is reported on the dashboard:
**`results/gpu/kaggle-t4-20261110-1753/`, commit `36e9026`**. That session's correctness gate
passed with 289 tests on the real GPU. An earlier passing session at `83a0fc7` (`…-1748`) gave the
same headline ratios within about ±0.03×.

**Read these numbers with the platform in mind.**
- *GPU:* Turing (sm_75), 40 SMs, 320 GB/s nominal memory bandwidth, 70 W board power. There is no
  bf16, so the bf16 paths report as unsupported.
- *Host:* a shared, virtualized Kaggle machine with 4 vCPUs (Xeon @ 2.0 GHz) and two T4s, of which
  the benchmarks use device 0. Its slow host CPU makes this setup **launch-bound**, and the results
  say more about that regime than about the T4's peak.
- *Software:* torch 2.11.0+cu130, Triton 3.6.0, driver 580.
- *Clocks:* sampled before and after each trial group, the SM clock was 1470–1590 MHz (median about
  1530) at 77 °C in P0, with occasional dips to 900 MHz.
- *Sample:* one session; each comparison has 20 interleaved, paired trials.

## Session history

| Run | Commit | Gate | Outcome |
|---|---|---|---|
| `kaggle-t4-20261110-1510` | `1e4076a` | failed (265 passed, 4 failed) | Found the CUDA-graph lifetime bug (below). No benchmarks ran |
| `diag-t4-20261110-1514`, `-1523` | | n/a | An on-GPU diagnostic isolated the cause, then verified the fix |
| `kaggle-t4-20261110-1556` | `79011c1` | failed (278 passed, 1 failed) | One test assumed no CUDA device; fixed in `83a0fc7` |
| `kaggle-t4-20261110-1748` | `83a0fc7` | **passed** (279) | First real NVIDIA numbers; they exposed four bugs, fixed in `36e9026` |
| `kaggle-t4-20261110-1753` | `36e9026` | **passed** (289) | Verified those fixes on the GPU; every step ok, Nsight included. **Reported** |

`gpulab analyze` reports measurements only from the latest session whose gate passed, so failed or
superseded sessions are listed but never pooled.

## Headline: fp16, 1024×2816×4, 8 rows, pipelined

| Implementation | Eager → candidate | Speedup [95% CI] | Weight streaming (derived) |
|---|---|---|---|
| eager + CUDA graph | 948 → 481 µs | **1.98× [1.96, 1.99]** | 73 → 144 GB/s (23% → 45% of nominal) |
| fused weights | 948 → 551 µs | **1.70× [1.67, 1.73]** | 126 GB/s (39%) |
| torch.compile | 948 → 594 µs | **1.55× [1.46, 1.60]** | 117 GB/s (36%) |
| compile (`reduce-overhead`) | 948 → 962 µs | 0.99× [0.98, 1.00] | n/a |
| **Triton fused kernels** | 948 µs → 2.34 ms | **0.38× [0.36, 0.41]: slower** | 30 GB/s (9%) |
| Triton + CUDA graph | 948 µs → 2.34 ms | 0.40× [0.38, 0.41]: slower | |

Bandwidth = 69.2 MB of fp16 weights ÷ measured time per forward; the 320 GB/s peak is the T4's
*theoretical* figure.

**Across the sweep** (1–64 rows; d_model 512, 1024 and 2048; fp16 and fp32): the CUDA-graph gain
shrinks as the model grows, from 2.35–2.59× at d=512 to 1.43–2.05× at d=1024 and 1.00–1.13× at d=2048. A
fixed launch cost behaves exactly like this: it gets amortized as the work per launch grows. Fused
weights gains 1.6–1.7× in fp16 and 1.1–1.3× in fp32. Triton is slower at every point, from 0.14×
to 0.62×.

## What the profiles show

**Eager is launch-bound.**
- *Nsight Systems*, eager at 8 rows: 60 kernels per forward, totaling 606 µs of kernel time. The
  pipelined forward takes 948 µs, so the GPU sits idle for about a third of it, waiting for the CPU
  to issue the next launch.
- *The PyTorch profiler* agrees: 60 kernels in a 1.87 ms GPU span. Profiler overhead widens the
  gaps further.
- *CUDA graphs* remove the per-launch CPU work, which is why the same kernels replayed as one graph
  are about 2× faster.
- *An unexplained detail:* the graph's 481 µs is less than eager's 606 µs of kernel time, so kernel
  durations themselves differ between the two modes. The clock samples are too coarse (taken only
  between trial groups) to say why.

**The Triton kernel is GPU-bound and slow.**
- *Launches are not the problem.* It issues 8 kernels per forward, against eager's 60, and
  CUDA-graph replay leaves it unchanged (0.38× → 0.40×).
- *Nsight Systems* (`cuda_graph_triton_r8_float16_cuda_gpu_kern_sum.csv`) puts the time inside the
  two kernels themselves:

  | Kernel, fastest instance per layer | Time | Effective bandwidth |
  |---|---|---|
  | `_norm_gate_up_silu` | 340 µs | about 34 GB/s |
  | `_down_residual` | 241 µs | about 24 GB/s |
  | cuBLAS GEMM in eager, average, for comparison | 34 µs | about 168 GB/s |

- *A caveat on that capture:* it includes Triton's autotuning, which shows up as 4,720 L2-flush
  `FillFunctor` kernels and as wide per-kernel spreads. The table therefore uses minimums, not
  averages. The minimums sum to about 2.3 ms per forward, which matches both the benchmark and the
  profiler.
- *Likely cause (a hypothesis, not yet confirmed with Nsight Compute): under-occupancy at small M.*
  - The grid has one program per output tile. At M = 8 that is 22–88 programs for gate/up and 8–32
    for the down projection, on a 40-SM GPU.
  - Each program streams the full K dimension serially, with no split-K.
  - Split-K, or stream-K, would raise parallelism. It needs a two-phase reduction, because the SiLU
    and residual epilogues apply after the full sum
    ([KERNELS.md](KERNELS.md#triton-kernels-nvidia)).
- *An unexplained gap:* in latency mode (a sync per call) Triton takes 4.0–7.2 ms against 2.3–7.0 ms
  pipelined. That gap is larger than eager's, and its cause is not established.

**`torch.compile(mode="reduce-overhead")` regresses as the model grows.** This is a diagnosis, not a
verified root cause.
- *The pattern:* 2.99× at d=512, 1.0× at d=1024, and 0.36–0.41× at d=2048.
- *What the profiles show:*
  - Its forward runs 23 kernels against plain `compile`'s 20. The extras are two
    `multi_tensor_apply` copy kernels and one more device-to-device memcpy.
  - Its kernel time is about 500 µs higher (956 vs 456 µs at 8 rows). Copying the 69 MB of weights
    once per replay would cost about that much at the T4's bandwidth.
- *The likely mechanism:* the weights are closure tensors (`src/gpulab/impls/compiled.py`), so
  Dynamo lifts them as graph inputs. Inductor's CUDA-graph trees then copy every input not known to
  be static into the graph's buffers on each replay.
- *The likely fix:* mark the weights static (`torch._dynamo.mark_static_address`), or hold them as
  module parameters or buffers.
- *Status:* not applied or measured yet.

## Serving (2 ms deadline, open-loop Poisson arrivals)

- *Eager:* service time is 1.13–1.22 ms against the 2 ms deadline. Dynamic batching holds the
  miss rate to 15–49% across 250–8000 rps; direct execution reaches 17–100%.
- *Triton:* its service time of 2.4–3.9 ms (up to 6.2 ms in fixed batches of 8) is already over the deadline, so it misses every
  deadline under every policy.
- *The autotune fix, verified:*
  - Before (`…-1748`): Triton's tuning was keyed by the exact row count, so every new batch size
    re-tuned in the middle of serving. Dynamic batching's p99 reached **12.7 s**.
  - After (`…-1753`): the key is bucketed to powers of two (`gpulab.kernels.tuning`), and the
    server's calibration warms every bucket. The worst dynamic-batching p99 is now **10 ms**.
- *Remaining outlier:* one request of 290 ms in `static8` at 8000 rps, against a p99 of 7.7 ms there.
- *Caveat:* every sweep point is a single short run, with no per-point CIs.

## Bugs that only real NVIDIA hardware exposed

1. **The CUDA-graph runner did not own what its graph reads.**
   - *What happened:* Triton's packed weight copy belonged only to the inner runner. Once that was
     garbage-collected, replays read freed memory.
   - *How it was found:* an on-GPU diagnostic disproved the cuBLAS-workspace hypothesis first.
   - *The fix:* `GraphRunner` owns the inner runner, the weights and the buffers.
   - *The test:* `test_graph_runner_owns_what_it_replays` overwrites freed memory with NaN before
     replaying, so it fails deterministically.
2. **The Triton autotune key followed the exact row count, causing multi-second serving stalls.**
   Fixed as described under Serving.
3. **Profiler kernel counts included annotation spans.**
   - *What happened:* the PyTorch profiler mirrors each `record_function` range onto the GPU
     timeline as a CUDA-typed event. Counting those as kernels added one fake "kernel" per forward,
     spanning the whole forward, so "kernel time" exceeded wall time.
   - *The fix:* `split_cuda_events` separates the spans; they are reported as `gpu_forward_span_us`.
     Profiles produced before the fix have their kernel fields withheld.
4. **The summary pooled every GPU session**, including ones whose gate failed. It now reports only
   the latest passing session.
5. **The Nsight script called a CLI path that does not exist on Kaggle.** It now resolves `gpulab`
   on `PATH`.
6. **The dashboard fetched data with `force-cache`**, so returning visitors kept stale results after
   a redeploy. It now uses `no-cache`.
7. **Provenance stored every GPU's driver line in each GPU's entry**, which showed up on the 2×T4 host. It now
   queries each device by `--id`.

## Not measured

- Nsight Compute: `ncu` is on the image but was not run.
- PTX/SASS inspection.
- bf16, which needs sm_80 or newer.
- Repeated sessions for between-session CIs.
- The `reduce-overhead` fix.
- Split-K.
