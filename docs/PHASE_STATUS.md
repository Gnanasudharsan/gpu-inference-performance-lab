# Phase status

| Phase | Title | Status | Commit |
|---|---|---|---|
| 0 | Research, repository and environment | **complete** | `2c42aa6`, `275775a` |
| 1 | Workload and correctness baseline | **complete** | see git log |
| 2 | Benchmarking and profiling | **complete** | `d82a86a` + docs/data commits |
| 3 | GPU kernel optimization | pending | |
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
