# Phase status

| Phase | Title | Status | Commit |
|---|---|---|---|
| 0 | Research, repository and environment | **complete** | `2c42aa6`, `275775a` |
| 1 | Workload and correctness baseline | **complete** | see git log |
| 2 | Benchmarking and profiling | pending | |
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

