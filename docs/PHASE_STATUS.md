# Phase status

| Phase | Title | Status | Commit |
|---|---|---|---|
| 0 | Research, repository and environment | **complete** | `2c42aa6`, `275775a` |
| 1 | Workload and correctness baseline | **complete** | see git log |
| 2 | Benchmarking and profiling | **complete** | `d82a86a`, `18e7cbf`, `fa7ac5f`, data commit |
| 3 | GPU kernel optimization | **complete** (Apple GPU measured; NVIDIA paths correctness-only) | `cb577af`→`8cff002`→`605be56`, phase commit |
| 4 | Scheduling and end-to-end systems | pending | |
| 5 | Experimental analysis | pending | |
| 6 | Web dashboard | pending | |
| 7 | Deployment and live execution | pending | |
| 8 | Final hardening and showcase | pending | |

## Phase 0 report

- Deliverables: repository skeleton, `gpulab env` provenance tool, CI (CPU tests plus the Triton
  interpreter job), research alignment, environment report, decisions, and limitations.
- Gate: clean install, local lint, types and tests passing, and CI green on GitHub.
- Result: **passed.** CI run on `275775a` was green: the CPU job, and the Triton 3.9.0 interpreter job
  with 3 kernel tests. The first CI run failed because the Triton job had no tests to select (pytest
  exit 5). This was fixed by adding a real interpreter smoke test.

## Phase 1 report

- Workload: RMSNorm → SwiGLU → residual stack ([WORKLOAD.md](WORKLOAD.md)), with deterministic
  weights and inputs that are identical across devices.
- Reference: float64 CPU, validated against independent NumPy.
- Baseline: `eager` implementation and `gpulab run` CLI. Execution modes are registered in
  `gpulab.impls`.
- Correctness: tolerances calibrated from measured baseline error on the Apple M4 CPU and GPU
  ([CORRECTNESS.md](CORRECTNESS.md)). The checker is shown to catch four injected bugs.
- Records: JSONL and Parquet with provenance. There are 18 baseline records in `results/phase1/`.
- Gate: 79 tests pass locally (CPU and MPS parametrizations). CI runs the CPU subset.


## Phase 2 report

- Harness modes: `latency` (sync per request; host and device separated), `e2e` (H2D/compute/D2H),
  and `throughput` (pipelined bursts). Also measured: build and first-call time, memory, deadline
  goodput, trial repeatability, bootstrap CIs with an autocorrelation check, and power/thermal state.
- `--interleave` rotates implementations trial by trial and computes paired speedups.
- Profiling: `gpulab profile` counts leaf dispatches per forward (64 for eager) and keeps top ops.
  Nsight scripts are prepared for NVIDIA hosts.
- Sample artifacts are in `results/phase2/` (JSONL, Parquet, REPORT.md, profiles, repeatability).
- Measured caveats on the M4 laptop (see BENCHMARK_METHODOLOGY): the latency tail is clustered (lag-1
  autocorrelation 0.5–0.77), cross-process p50 varies by up to 2.2×, and pipelined (throughput-mode)
  timing is far more stable (trial CV ≈ 0.08).
- Gate: 94 tests pass (stats, harness timing against a known 2 ms delay, correctness gate,
  interleaving, profiler scaling, CLI round trip). Repeatability is measured and documented.

## Phase 3 report

- Bottleneck first: eager issues 64 launches per forward, of which 12 are GEMMs, and streams weights
  at 55–66% of nominal bandwidth at 1–8 rows.
- Implementations: `fused_weights`, `compile`, `metal` (fused GEMV kernels, v1→v2 with an interleaved
  A/B, tuned launch configs), `triton` (fused tiled GEMMs), and the CUDA-graph variants.
- Correctness: every implementation is checked against the float64 reference on all supported
  devices and dtypes (210 tests locally). Triton fp32/fp16 pass in the CI interpreter; bf16 is blocked
  by an interpreter bug that a minimal diagnostic reproduces.
- Measured (pipelined, fp16, 1024×2816×4): metal is 1.16–1.26× faster than eager at 1–8 rows, with
  launches cut 64→8 and bandwidth 55%→68% of nominal. At 16 rows there is no significant
  difference. At d=512 it is 1.66×.
- Regressions documented: CPU `torch.compile` (0.63–0.78×), CPU `fused_weights` at 8 rows (0.79×),
  and one MPS fp32 latency-mode case that the pipelined mode contradicts.
- Not measured: anything CUDA-specific (no NVIDIA GPU).
- Gate: correctness passes; speedups and regressions are backed by paired CIs; profiler launch counts
  explain the gains. Process bugs found and fixed on the way: the interleaved CLI silently skipped
  throughput mode, the profiler missed custom kernels, and lint failures were masked by pipes.
