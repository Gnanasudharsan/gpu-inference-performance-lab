# Phase status

| Phase | Title | Status | Commit |
|---|---|---|---|
| 0 | Research, repository and environment | **complete** | `2c42aa6`, `275775a` |
| 1 | Workload and correctness baseline | **complete** | see git log |
| 2 | Benchmarking and profiling | **complete** | `d82a86a`, `18e7cbf`, `fa7ac5f`, data commit |
| 3 | GPU kernel optimization | **complete** (Apple GPU measured; NVIDIA paths correctness-only) | `cb577af`→`8cff002`→`605be56`, phase commit |
| 4 | Scheduling and end-to-end systems | **complete** | `758dbd6`, phase commit |
| 5 | Experimental analysis | **complete** | `10828f8`, phase commit |
| 6 | Web dashboard | **complete** | phase commit |
| 7 | Deployment and live execution | **complete for the dashboard**; GPU API prepared, not deployed (needs authorization) | phase commit |
| 8 | Final hardening and showcase | **complete** | release commit |

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

## Phase 4 report

- Serving layer: deterministic traces, open-loop single-threaded replay, direct/static/dynamic
  policies, bounded queue, expired-request dropping, bucket padding, failure isolation, and
  replay-fidelity (wake-lateness) metrics. 22 serving tests, stress-run for flakiness.
- Bugs found by tests and fixed: slack scheduling without a margin missed deadlines because of
  wake-up jitter; macOS sleep overshoot (up to ~10 ms) corrupted replay fidelity, so spin-wait is
  now the default and lateness is always reported.
- Measured: dynamic batching gives 4–5× the peak goodput of direct execution and the lowest miss rate
  under bursts. Static batching fails at low load. The operating envelope is narrow, because of
  measured 3–10 ms GPU round-trip stalls in ~7% of requests. The idle-gap penalty is confirmed with
  paired CIs.
- Gate: reproducible replay (seeded traces), load sweep with p99, documented envelope. Caveat:
  each sweep point is one run.

## Phase 5 report

- Runner: `gpulab experiment <config.toml>`, with configs `smoke` (CI), `apple_m4` (full local
  matrix), `repro_check`, and `nvidia` (prepared, not executed). Skipped work is recorded with a
  reason.
- Analysis: `gpulab analyze` produces `results/summary.json` (the evidence-labeled dataset for the
  dashboard) and `results/ANALYSIS.md` (114 paired comparisons with CIs, 5 regressions, roofline,
  serving, envelope, idle gap, launch counts, correctness).
- Reproduction check: an independent re-run through the runner replicated the Phase 3 headline
  speedups within overlapping CIs (`results/phase5/repro_comparison.json`).
- Narrative: [ENGINEERING_REPORT.md](ENGINEERING_REPORT.md), covering bottlenecks, wins, failures,
  constraints, evidence categories and next steps.

## Phase 6 report

- A static Next.js dashboard with 9 sections: overview, live execution, benchmark explorer, latency
  analysis, serving, kernel analysis, engineering decisions, reproducibility, architecture.
- Live execution goes through a bounded FastAPI service (`gpulab serve`). It was verified from the
  browser against the local Apple M4 backend: a 1.25× [1.19, 1.35] paired result, labeled LIVE.
- The palette was validated (CVD, contrast) against the real surface. Accessibility and responsive
  checks were run on every page ([DASHBOARD.md](DASHBOARD.md)). Screenshots of the real app are in
  `docs/screenshots/`.
- Gate: production build succeeds (CI job added); no fabricated live state; recorded and live data
  are kept separate.

## Phase 7 report

- Dashboard deployed to https://gpu-inference-performance-lab.vercel.app (static, security headers,
  verified with HTTP checks; the Live page reports *offline* without a backend).
- GPU execution API: hardened (auth, rate limit, one job at a time, bounded inputs, CORS including
  Private Network Access, hashed client ids, JSON logs). Packaged as `deploy/Dockerfile` (base image
  tag verified) and `deploy/modal_app.py` (`max_containers=1`, timeout, scale-to-zero, secret
  store). **Not deployed: no GPU-cloud credentials and no authorized spend.**
- A cross-origin live run from the production origin to a local M4 API was verified at the HTTP
  level (preflight plus run).
- See [DEPLOYMENT.md](DEPLOYMENT.md) for the hosting evaluation, cost caps and the exact checklist.

## Phase 8 report

Hardening checks and their results are in [final-audit.md](final-audit.md):
- tests and the Triton interpreter suite
- the GPU performance regression gate (passed)
- static analysis with security rules
- dependency audits (0 known vulnerabilities)
- a secrets scan
- an API load and fuzz test (passed)
- accessibility, responsiveness, documentation links and a deployment smoke test

The README was rewritten with verified results only. The interview walkthrough and final audit,
including resume bullets limited to measured outcomes, are in `docs/`.

## Post-release: NVIDIA validation path (Kaggle T4)

- Chosen platform: the Kaggle free tier (T4), which needs no sponsor and is scriptable end to end.
  Explorer is the upgrade path.
- Added `scripts/gpu_session.py`, a portable session driver with provenance, a capability gate, a
  correctness gate that blocks benchmarks on failure, the experiment, CUDA kernel profiles, optional
  Nsight and an archive. Verified locally in dry-run mode, including the gate-failure abort.
- Added `experiments/nvidia_t4.toml`, `deploy/kaggle/` and `scripts/kaggle_run.sh` (push, run with a
  3 h cap, download, ingest).
- The Triton bf16 path is reported unsupported below compute capability 8.0.
- Ingestion: `gpulab analyze` merges `results/gpu/*`. The dashboard gains CUDA views only when real
  CUDA records exist; the layout was checked once with a scratch-only fixture, which was deleted.
- **Pending:** the run itself, which needs the owner's Kaggle API token.
