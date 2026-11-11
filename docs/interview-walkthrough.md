# Interview walkthrough

All numbers were measured on an Apple M4 GPU (Metal/MPS) unless labeled otherwise. Do not
extrapolate them to NVIDIA hardware.

## 5-minute version

1. **Problem (30 s).** Low-latency inference for models smaller than LLMs, at 1–8 rows per request.
   The block is RMSNorm → SwiGLU → residual (the feed-forward block of most modern networks), with
   four layers at 1024×2816.
2. **Diagnosis (60 s).** At 1–8 rows a forward does ~1–8 FLOPs per byte, so the floor is weight bytes
   ÷ bandwidth (69 MB ÷ 120 GB/s ≈ 0.58 ms, theoretical). Eager took ~0.9–1.05 ms. The profiler showed
   **64 launches per forward, only 12 of them GEMMs**, plus intermediates round-tripping through
   memory. At 8 rows eager streamed weights at 55% of nominal bandwidth.
3. **Fix (90 s).**
   - Two fused GPU kernels per block.
   - RMSNorm is folded in by linearity: `(s·x⊙g)W = s·((x⊙g)W)`, with the per-row scale applied in
     the epilogue.
   - Each weight row is streamed exactly once with coalesced vector loads, and reduced with
     `simd_sum`.
   - SiLU·mul and the residual are fused into epilogues.
   - Result: launches 64 → 8, and **1.16–1.26× faster at 1–8 rows** with every 95% CI above 1.
     Bandwidth went 55% → 68% of nominal at 8 rows.
4. **Rigor (60 s).** Separate runs on this laptop differed by up to 2.2×, more than the effect itself.
   So every comparison is interleaved (rotated order per trial), paired, and bootstrapped. The result
   replicated three times: 8 rows gave 1.205×, 1.210× and 1.209×.
5. **System lesson (60 s).** Under open-loop serving, dynamic batching gave 4–5× the goodput of direct
   execution. The kernel speedup barely moved serving, though: each request costs ~2 ms of host↔GPU
   round trip against ~0.8 ms of compute. The next lever is the round trip, not the kernel.

## 20-minute version

### 1. Framing (2 min)

- **What latency-sensitive means here.** Deadlines in the milliseconds, batch sizes near 1, and
  goodput (on-time work) rather than throughput.
- **What I could and could not measure.** The kernels were developed on an M4. The NVIDIA paths
  were then measured on a free Kaggle T4 behind a correctness gate.
  - *What it showed:* the host was launch-bound, so CUDA graphs won (1.98×), and my Triton kernel
    lost to cuBLAS (0.38×). I report both.
  - *What the hardware found:* seven bugs that CPU-only CI could not see. The best story is the
    CUDA-graph memory-lifetime bug ([NVIDIA_T4.md](NVIDIA_T4.md)).

### 2. Correctness first (3 min)

- **The reference.** A float64 CPU reference, validated against an independent NumPy implementation.
- **The tolerances.** Scale-normalized errors (relative L2, max error ÷ max |ref|), with tolerances
  calibrated at 3–10× the eager baseline's *measured* error per dtype.
- **The checker is tested too.** It must catch four injected bugs: swapped gate/up, a dropped
  residual, a wrong ε, and permuted rows.
- **The harness enforces it.** It refuses to time anything that fails correctness.
- **Triton in CI.** The kernels run under `TRITON_INTERPRET=1`. bf16 failed with errors around 1e11.
  A minimal two-tile `tl.dot` reproduced it, so it is an interpreter bug, now a strict xfail.

### 3. Measurement (4 min)

- **Three timing modes:**
  - *synchronous latency*: what a caller waits
  - *pipelined bursts*: the cost of the work itself
  - *e2e*: H2D, compute and D2H, with stage breakdown
- **Biases I found, and their fixes:**
  - *Device-event instrumentation perturbed host latency on MPS*, so device time is measured in a
    separate pass.
  - *A 0.25 s warm-up was too short* for clocks to ramp.
  - *The latency tail is clustered* (lag-1 autocorrelation up to 0.8), so i.i.d. bootstrap CIs are
    flagged unreliable when it is high.
  - *Separate runs disagree* by up to 2.2×, so comparisons interleave.
- **A hypothesis I rejected.** I suspected stalls quantized to the 120 Hz display frame (8.3 ms). The
  histogram showed a smooth 3–10 ms decay, so the hypothesis was rejected and the cause is left
  undetermined.

### 4. The kernel (5 min)

- **Layout.** One simdgroup of 32 lanes owns a few output columns, which are contiguous weight rows.
  Lanes stride the reduction with `half4` loads, accumulate in fp32, and reduce with `simd_sum`. There
  are no barriers in the hot loop.
- **The norm by linearity.** Each threadgroup recomputes `rsqrt(mean(x²)+ε)` for its rows; `x` is a
  few KB and cache-resident. This saves a pass and a launch. Rounding differs from eager, but stays
  within tolerance.
- **v1 → v2.**
  - *What broke:* v1 regressed to **0.77× eager at 8 rows**. The column loop sat outside the K-loop,
    so `x` was reloaded per column (~4× the weight loads), and runtime-bounded accumulator arrays
    likely spilled.
  - *The fix:* v2 specializes per row count (accumulators in registers) and puts the column loop
    inside the K-loop.
  - *The result:* **1.67× over v1** at 8 rows. The A/B loaded v1 from Git history, interleaved.
- **Tuning.** A 12-config sweep, measured interleaved. 4 columns × 8 rows was 27–30% slower, likely
  register pressure. I didn't profile further, so I say "likely".
- **The crossover.** At 16 rows the GEMV formulation is wrong, and vendor GEMM wins. The path hands
  back there, and the measured speedup is 1.00× (n.s.).

### 5. Serving (3 min)

- **The replay.** Open-loop and single-threaded discrete-event, so there is no GIL-fighting client
  thread. macOS sleep overshot by up to ~10 ms, so the replay spin-waits and reports its own wake-up
  lateness (p99 ~0.1 ms).
- **Policies:**
  - *direct* saturates at ~400–600 rps
  - *static-8* misses every deadline below 300 rps (8 ÷ rate ≫ 10 ms)
  - *dynamic* batching, with a p90 service prediction and a slack margin, gives the highest goodput
    and the lowest miss rate under bursts
- **Two corrections I made to my own write-up:**
  - The p99 outliers above 100 ms were rare single-batch GPU stalls of 114–392 ms, not queue backlog.
    Queueing never exceeded the deadline.
  - The 1% miss budget is unattainable above ~100 rps on this machine, because of round-trip stalls.

### 6. What I'd do next (3 min)

1. On one NVIDIA GPU, under a budget, collect:
   - Nsight Systems timelines, eager vs `cuda_graph_triton`, to see launch gaps disappear
   - Nsight Compute DRAM throughput for the Triton kernels at M ≤ 16
   - a check of whether split-K is needed for the narrow down projection
2. Pipeline the serving round trip: double-buffered H2D/compute/D2H, or device-resident inputs.
3. Use int8 or fp8 weights. The floor is weight bytes, so this is the biggest lever left, but it
   needs an accuracy study.

## Likely questions

- **Why not just use `torch.compile`?** I did measure it. It was neutral on MPS for this workload,
  because the GEMMs dominate and stay vendor GEMMs, and a regression on CPU. Fusion inside the
  product needs a custom kernel.
- **Is 1.2× meaningful?** It is a paired, interleaved, CI-bounded result that replicated three times.
  It is close to what the bandwidth gap allows: 68% of the nominal peak is achieved, and that peak is
  rarely reachable in practice.
- **How would CUDA graphs change things?** On NVIDIA they remove per-kernel CPU launch cost, and the
  implementation is ready (static buffers, rejection of other row counts, bucket padding in serving).
  I have no measurement yet, so I won't guess a number.
