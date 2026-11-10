# Engineering decisions

Each entry records what was decided, why, the alternatives that were rejected, and what would change
the decision. Newest entries go at the bottom.

## D1. Workload: a stack of pre-norm RMSNorm → SwiGLU gated-MLP → residual blocks

**Decision.** Each block computes `y = x + W_down( silu(W_gate · n) ⊙ (W_up · n) )`, where
`n = RMSNorm(x)`. A network is `depth` such blocks. Batch sizes are 1–64 rows; hidden widths are
256–2048 with a 2.67–4× expansion.

**Why.** At small batch, the GEMMs have arithmetic intensity of roughly `batch` FLOP/byte of weight
traffic, so they are memory-bandwidth-bound. Eager execution issues about 8–10 kernels per block
(the norm's reduction and scale, three GEMMs, SiLU, multiply, add). That makes launch overhead and
intermediate memory round-trips first-order costs, the same regime the public sources describe for
low-latency inference with models smaller than LLMs. The block also appears in most modern transformer
and MLP stacks, so the optimizations transfer.

**Rejected.** A full transformer (attention adds KV-cache and sequence-length dimensions that dilute
the low-latency story and are already covered by mature libraries). A plain ReLU MLP (too little
fusable structure). CNNs (not representative of tabular or sequential inference at small batch).

## D2. Development hardware is Apple M4; NVIDIA results are a separate, explicitly gated step

**Decision.** Measure everything possible on the Apple M4 CPU and GPU. Implement CUDA-only paths
(Triton kernels, CUDA graphs) and validate their correctness with the Triton CPU interpreter in Linux
CI. Mark their performance as *pending* until an authorized NVIDIA run happens.

**Why.** It is honest and reproducible on available hardware. The Apple GPU is a real GPU with real
launch overhead, memory bandwidth limits and a custom-kernel path (Metal via
`torch.mps.compile_shader`), so the methodology is exercised end to end.

**Rejected.** Simulating or estimating CUDA numbers (that would be fabrication). Renting a GPU without
authorization (the operating principles forbid it).

## D3. Python tooling: uv, hatchling, ruff, mypy, pytest

Fast, standard, and small on disk, which matters because only about 7 GB is free. No Poetry or conda.

## D4. CI runs on CPU-only PyTorch, plus a separate Triton-interpreter job

The Triton interpreter (`TRITON_INTERPRET=1`) runs `@triton.jit` kernels on CPU tensors with NumPy
semantics. It checks indexing, masking and numerics, but not performance, and it does not catch every
GPU-only failure (for example, shared-memory limits or register pressure). Those are covered by the GPU
validation plan.

## D5. Golden reference is float64 on CPU, not the eager implementation

Comparing optimized kernels only against eager in the same dtype would hide errors that both share,
and it would make the reference depend on the device. A float64 CPU reference is independent of all
implementations under test. The reference itself is cross-checked against a separate NumPy
implementation.

## D6. Scale-normalized error metrics with calibrated per-dtype tolerances

`torch.allclose`-style `atol` and `rtol` values are sensitive to activation magnitude and to elements
near zero. Normalizing by ‖ref‖ and max|ref| gives thresholds that hold across shapes and depths.
Tolerances were set at 3–10× the measured baseline error ([CORRECTNESS.md](CORRECTNESS.md)).

## D7. Experiment records are JSON Lines with an `evidence` field

Every record carries `evidence ∈ {measured, derived, theoretical}` and a full provenance block
(hardware, versions, Git commit and dirty flag, hashed hostname). JSONL is append-only and
diff-friendly. Parquet is produced from it for analysis.

## D8. Host latency and device time are measured in separate passes

Measured on MPS: recording a pair of events around every request raised host p50 and variance.
Host latency is the user-facing number, so it is measured without instrumentation. Device time
comes from a second pass and is never subtracted sample-by-sample from host time.

## D9. Comparisons are interleaved and paired

Cross-process p50 on the M4 varied by up to 2.2× (`results/phase2/repeatability_mps.json`), which
is larger than many optimizations. Comparing separately run numbers would be meaningless here.
`run_interleaved` rotates the implementation order every trial, and speedup is the geometric mean
of per-trial ratios, reported with its min and max across trials.

## D10. Pipelined (throughput-mode) time is reported alongside synchronous latency

Synchronous per-request latency on this laptop is dominated by power-state effects (clustered
tails). Pipelined time per request is much more stable and isolates the cost of the work itself.
Both are reported, and neither is substituted for the other.

## D11. Custom kernels target only the regime where they win (≤ 8 rows)

The fused Metal GEMV kernels beat eager at 1–8 rows. At 16+ rows MPS's GEMM (simdgroup-matrix
hardware) is the better tool for the products, so `metal` switches to it and keeps only fused
elementwise kernels. The crossover was measured (16 rows: no significant difference), not assumed.
**Rejected:** writing a general Metal GEMM. Competing with Apple's GEMM at large M is a large
project with little payoff for a latency-oriented workload.

## D12. Kernel iterations are compared head-to-head from Git history

v1 and v2 of the Metal kernels were A/B tested in one interleaved run. v1 was loaded from its commit
with `git show`, so the comparison stays reproducible without keeping dead code in the package.

## D13. CUDA C++ / PTX not written

For this workload, Triton expresses the needed fusion (norm folded into the K-loop, two accumulators,
nonlinear epilogues) at the tile level. No NVIDIA hardware was available to justify or validate
hand-written CUDA or PTX. **Revisit** if Nsight Compute on a real GPU shows Triton failing to reach
DRAM bandwidth at small M (for example, if split-K is needed).

## D14. Launch-config tuning and comparisons use pipelined mode

Synchronous single-request latency on this laptop is dominated by clustered power-state noise
(Phase 2). Pipelined bursts isolate the cost of the work, and their CIs are 3–5× tighter. Synchronous
latency is still measured and reported for every comparison.

## D15. Open-loop, single-threaded discrete-event replay

A closed-loop client hides queueing. A separate client thread fights the GIL and adds up to 5 ms of
switch-interval jitter. Admitting arrivals at each scheduling point with their true timestamps is
exact for a non-preemptive single-device server.

## D16. Spin-wait while idle (configurable)

Measured: macOS `time.sleep` woke up as much as ~10 ms late under background QoS, which inflated
queueing delays. Spinning had p99 lateness of ~0.1 ms and did not slow GPU service (spin vs sleep:
service p50 2.0–2.1 ms vs 2.1–2.6 ms). Spinning costs one CPU core, so `idle_wait="sleep"` is kept
as an option, and lateness is reported for every run.

## D17. Dynamic batching uses a p90 service prediction and an explicit slack margin

Waiting until exactly *deadline − median service* missed deadlines in testing, through service
variance and wake-up jitter. A p90 prediction plus a 300 µs margin is the conservative default.
