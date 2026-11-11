# Final audit

A skeptical review of the repository as of the release commit, written to separate what is
verified from what is not.

## Checks run (Phase 8)

| Check | Result | Evidence |
|---|---|---|
| Unit + integration tests | **228 passed**, 199 skipped (device- or toolkit-specific) locally; CI CPU job green | `make check`, GitHub Actions |
| Triton kernels (interpreter) | **14 passed**, 4 skipped (bf16, reason recorded), 1 strict xfail (interpreter bf16 bug) | CI `triton-interpreter` job |
| Numerical correctness | all implementations within calibrated tolerances on every supported device×dtype; checker catches 4 injected bugs | `tests/test_impls.py`, `tests/test_correctness.py` |
| GPU performance regression gate | **passed**: metal vs eager now 1.116× (1 row) and 1.209× (8 rows), both "faster", both above their floors | `results/phase8/perf_regression.json` |
| Static analysis | ruff (incl. bandit `S` rules, triaged) and mypy (`disallow_untyped_defs`): clean | `make lint typecheck` |
| Dependency audit | `pip-audit`: no known vulnerabilities; `npm audit --omit=dev`: 0 vulnerabilities | Phase 8 run |
| Secrets scan | no tokens, keys or private keys in the tracked tree or the deployed bundle | `git grep` patterns + bundle scan |
| API load and robustness | 200/200 health checks under 32-way concurrency; concurrent runs → exactly 1 measurement + 429s; 11 malformed or hostile inputs → 4xx only | `results/phase8/api_load_test.json` |
| Frontend accessibility | 9/9 pages: single h1, lang, labeled controls, named buttons, labeled charts, table views, contrast ≥ 4.5:1 | [DASHBOARD.md](DASHBOARD.md) |
| Responsive | no horizontal page scroll at 375 px and 1280 px | browser checks |
| Documentation links | all relative links resolve | link checker |
| Deployment smoke test | production `/`, `/live/`, `/data/summary.json` → 200; unknown → 404; security headers present | `scripts/deploy_frontend.sh` |

## Verified results (measured or derived from measurements)

- **Fused Metal kernels vs eager.** Pipelined, fp16, 1024×2816×4, Apple M4: 1.16× [1.11, 1.23],
  1.21× [1.16, 1.26], 1.26× [1.22, 1.30] and 1.20× [1.14, 1.27] at 1, 2, 4 and 8 rows. At 16 rows:
  no significant difference.
- **Launches per forward:** 64 → 8 at ≤ 8 rows.
- **Bandwidth at 8 rows:** 65.8 → 81.9 GB/s, which is 55% → 68% of the nominal 120 GB/s (the peak is
  theoretical).
- **Smaller model:** 1.66× [1.52, 1.81] at 512×1408×4, 1 row.
- **Kernel iteration:** v2 is 1.67× over v1 at 8 rows; v1 was 0.77× eager.
- **Replication:** 8 rows gave 1.205×, 1.210× and 1.209× across three independent runs.
- **Serving (10 ms deadline).**
  - *Peak goodput:* dynamic batching reached 1,503 rows/s (compile) and 1,114 rows/s (metal), against
    a direct-execution peak of 247–324 rows/s.
  - *Bursty arrivals:* the dynamic miss rate was 13.5–19.5%, against 57–69% for direct.
- **Idle-gap penalty:** a 2 ms idle gap made requests 1.19× [1.01, 1.39] and 1.35× [1.18, 1.53]
  slower in two paired runs.
- **NVIDIA T4 (Kaggle, one session, correctness gate passed with 289 tests on the GPU).** fp16,
  1024×2816×4, 8 rows, pipelined:
  - *eager + CUDA graph:* 1.98× [1.96, 1.99]
  - *fused weights:* 1.70× [1.67, 1.73]
  - *torch.compile:* 1.55× [1.46, 1.60]
  - *Triton fused kernels:* 0.38× [0.36, 0.41], a measured loss
  - *Replication:* a second passing session agreed within about ±0.03×.
  - *Details:* [NVIDIA_T4.md](NVIDIA_T4.md).
- **Live API from the browser:** 1.25× [1.19, 1.35] in a live run, and 1.28× [1.18, 1.41] in a
  cross-origin run from the production origin.

## Known limitations and incomplete work

1. **NVIDIA coverage is one free-tier T4.**
   - *Measured:* Triton, CUDA graphs, `reduce-overhead` and Nsight Systems.
   - *Missing:* Nsight Compute, PTX/SASS, bf16, data-center GPUs, and between-session CIs.
   - *Open questions:* the cause of the Triton slowdown (occupancy is a hypothesis), and the
     `reduce-overhead` fix (diagnosed, not measured).
2. **The GPU execution API is not deployed**; the dashboard's Live page needs a user-run backend.
   `deploy/Dockerfile` has not been built (no Docker daemon locally).
3. **Triton bf16** is unverified: the CPU interpreter has a bf16 bug, and the T4 has no bf16.
4. **Development machine noise.**
   - Synchronous latency on the laptop GPU has clustered 3–10 ms stalls (~7% of round trips), and
     rare single-batch stalls of 114–392 ms.
   - Absolute p99 values are environment-dominated.
   - Paired speedups are the robust quantity.
5. **Serving sweep points are single 3-second runs.** There are no per-point CIs, and the queue bound
   (256) was not tuned to the deadline.
6. **`torch.compile` launch counts are incomplete**: Inductor kernels are invisible to the counting
   method.
7. **No research report.** The supplied Deep Research report was not available. The alignment uses
   public primary sources ([research-alignment.md](research-alignment.md)).
8. **Single device only**: no distributed training or collectives, which are out of scope.

No known critical defects remain. Every defect found during development was fixed and is visible in
the Git history:
- the interleaved CLI silently skipped modes
- the profiler missed custom kernels
- a porcelain parsing bug in provenance
- wake-up jitter in the replay
- a float-ulp test flake
- a mis-attributed p99 claim in the docs
- seven bugs that only real NVIDIA hardware exposed:
  - the CUDA-graph runner did not own what its graph reads
  - Triton re-autotuned in the middle of serving (p99 12.7 s)
  - profiler annotation spans were counted as kernels
  - the summary pooled failed sessions
  - the Nsight script used the wrong CLI path
  - the dashboard served a stale cached dataset
  - the driver was duplicated in provenance

  Details are in [NVIDIA_T4.md](NVIDIA_T4.md#bugs-that-only-real-nvidia-hardware-exposed).

## Follow-up opportunities (by expected value)

1. Make the Triton kernel competitive: Nsight Compute on the T4 to test the occupancy hypothesis,
   then split-K or stream-K. Then a data-center GPU (L4 or H100) run.
2. Apply and measure the static-weights fix for `reduce-overhead`.
3. Pipelining the serving round trip (double-buffered H2D/compute/D2H, or device-resident inputs).
4. int8/fp8 weights with an accuracy study.
5. Computing the norm statistic once per forward.
6. Repeated serving runs with CIs, a deadline-derived queue bound, and a stall watchdog with hedging.

## Resume bullets (measured, verifiable from this repository)

- Built a reproducible GPU inference performance lab (Python/PyTorch, Metal, Triton, FastAPI,
  Next.js). Every result carries hardware and Git provenance, and every comparison is interleaved
  with a paired 95% bootstrap CI.
- Wrote fused Metal GEMV kernels for small-batch RMSNorm→SwiGLU→residual inference. They cut kernel
  launches per forward from 64 to 8 and ran 1.16–1.26× faster than eager PyTorch at 1–8 rows on an
  Apple M4 GPU, lifting weight-streaming bandwidth from 55% to 68% of nominal peak.
- Diagnosed and fixed a kernel regression (0.77× eager at 8 rows), using profiling and an
  interleaved A/B against the previous commit. The rewrite was 1.67× faster than the first version.
- Designed a measurement methodology robust to a noisy GPU: separate runs differed by up to 2.2×, yet
  paired interleaved trials replicated the headline speedup three times (1.205×, 1.210×, 1.209×).
- Implemented a latency-aware serving layer with open-loop replay, bounded queues and deadline-aware
  dynamic batching. It reached 4–5× the peak goodput of direct execution under a 10 ms deadline, and
  showed that host↔GPU round trips, not kernels, bounded end-to-end latency.
- Measured the same workload on an NVIDIA T4. The host was launch-bound (Nsight: 606 µs of kernels
  in a 948 µs forward), so CUDA-graph replay gave 1.98× [1.96, 1.99] over eager PyTorch. That beat
  operator fusion (1.70×) and torch.compile (1.55×).
- Found and fixed seven bugs that only real NVIDIA hardware exposed. One was a CUDA-graph
  memory-lifetime bug, isolated with an on-GPU diagnostic and pinned by a deterministic test.
  Another was Triton re-autotuning per batch size in the middle of serving, which cut p99 from
  12.7 s to 10 ms once fixed.
- Shipped a static observability dashboard and a bounded, authenticated, rate-limited live-benchmark
  API with load, fuzz, dependency and accessibility checks in CI.

*Not claimed:* a Triton speedup (it is a measured 0.38× slowdown on the T4), any data-center GPU
number, or a verified cause for the Triton slowdown.
