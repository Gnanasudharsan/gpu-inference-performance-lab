# Research alignment

This document ties the project's technical scope to publicly verifiable information about Jane Street's
Machine Learning Performance Engineer role. It does **not** claim knowledge of Jane Street's internal
systems. Anything not directly stated by a primary source is labeled *inference*.

## Input gap

The brief for this project said a Deep Research report would be supplied. No report was attached to the
request or present in the repository when work started (2026-11-10). This document was therefore compiled
from primary sources fetched on 2026-11-10. If the report is supplied later, reconcile it here and note
any change of scope in [DECISIONS.md](DECISIONS.md).

## Primary sources (fetched 2026-11-10)

| # | Source | Type |
|---|--------|------|
| S1 | [ML Performance Engineer, New York](https://www.janestreet.com/join-jane-street/position/7449077002) (same text: [London](https://www.janestreet.com/join-jane-street/position/7449252002), [Hong Kong](https://www.janestreet.com/join-jane-street/position/7449190002)) | Job posting |
| S2 | [Signals and Threads ep. 21: "The Uncertain Art of Accelerating ML Models"](https://signalsandthreads.com/the-uncertain-art-of-accelerating-ml-models/) (Sylvain Gugger, 2024-10-14) | Podcast |
| S3 | [Tech talk: "Making GPUs Actually Fast: A Deep Dive into Training Performance"](https://www.janestreet.com/tech-talks/making-gpus-actually-fast) | Talk |
| S4 | [Signals and Threads ep. 22: "Finding Signal in the Noise"](https://signalsandthreads.com/finding-signal-in-the-noise/) (2025-03-10) | Podcast |
| S5 | [Blog: "Trading off compute for memory with activation checkpointing"](https://blog.janestreet.com/trading-off-compute-for-memory-with-activation-checkpointing/) (2026-09-30) | Blog |
| S6 | [Blog: "Flex Model: an event-driven feature engineering system"](https://blog.janestreet.com/flex-model-an-event-driven-feature-engineering-system/) (2026-10-13) | Blog |

Two URLs returned 404 at fetch time (a second posting ID and a GPU-hackathon event page). Claims that
appeared only in search snippets for those pages, such as specific GPU SKUs, are not used here.

## Verified findings

- **Scope of the role (S1).** The role optimizes model performance for training and inference across three
  workloads: large-scale training, *low-latency inference in real-time trading systems*, and
  high-throughput inference for research.
- **Whole-system view (S1).** The posting asks for a whole-systems approach across storage, networking,
  host and GPU, and it asks explicitly whether throughput is really *goodput*.
- **Named low-level intuitions (S1).** The posting names the latency and throughput of CUDA graph launch,
  Tensor Core arithmetic, warp-level synchronization and asynchronous memory loads.
- **Named tools (S1).** CUDA, PTX/SASS, Triton, CUTLASS, cuBLAS, cuDNN, CUB/Thrust, Nsight Systems and
  Nsight Compute, CUDA-GDB, NCCL/MPI, InfiniBand/RoCE/NVLink.
- **Escalation order (S3).** The talk describes working in plain PyTorch first, then Triton, then CUDA,
  then PTX. It also says the team usually uses PyTorch.
- **Typical bottlenecks (S2, S3).** These are hidden CPU–GPU synchronizations, launch overhead (addressed
  with CUDA graphs), memory-bound elementwise work (addressed with fusion via `torch.compile` or Triton),
  and data loading. The three bottleneck classes are compute, memory and overhead.
- **Model size (S2, S4).** Models are smaller than frontier LLMs, so LLM-centric designs may not transfer.
  Latency needs range from roughly 100 ns to milliseconds, served across CPUs, FPGAs and GPUs.
- **Train/serve parity (S6).** The same feature code runs live and in historical replay, and the hot path
  avoids allocation to protect tail latency.

## Inferences (hypotheses, not facts)

- The GPU part of real-time inference probably runs at batch sizes near 1 and in the microsecond-to-
  millisecond range, where **launch overhead and memory bandwidth, not FLOPs**, dominate. This project's
  workload is chosen to sit exactly in that regime.
- The very small (~100 ns) budgets are probably served by CPUs or FPGAs rather than GPUs.
- Goodput, as used in the posting, likely covers both training (useful steps per unit time) and inference
  (requests that finish inside their deadline). This project uses the inference definition.

## Technical priorities derived from the sources

1. Measure first. Separate kernel time, launch overhead, host orchestration, transfers and queueing.
2. Remove launch overhead for stable shapes (CUDA graphs on NVIDIA; command-buffer batching on Metal).
3. Fuse memory-bound epilogues: norm, activation and gating, and the residual add.
4. Escalate deliberately: eager PyTorch, then algebraic fusion, then `torch.compile`, then custom kernels
   (Triton on CUDA, Metal on Apple GPUs), and document where each step stops paying off.
5. Report tail latency and deadline goodput, not only peak throughput.
6. Keep every optimized path bit-for-bit traceable to a reference with explicit tolerances.

## Skills-to-project matrix

| Skill named or implied by the sources | Where this project demonstrates it | Evidence type |
|---|---|---|
| Low-latency inference (S1) | Small-batch gated MLP workload; latency-SLO serving layer (Phase 4) | Measured (CPU and M4 GPU) |
| CUDA graph launch latency (S1, S2) | CUDA-graph execution mode (Phase 3) | Implemented; measurement needs NVIDIA GPU |
| Triton (S1, S3) | Fused RMSNorm and fused GEMM+SwiGLU / GEMM+residual Triton kernels | Correctness via `TRITON_INTERPRET=1` in CI; perf needs NVIDIA GPU |
| Custom GPU kernels on available hardware | Fused Metal kernels via `torch.mps.compile_shader` on Apple M4 | Measured |
| Fusion and `torch.compile` (S2, S3) | `fused_weights` and `compile` modes compared with eager | Measured |
| Nsight Systems / Compute (S1, S3) | Profiling scripts and methodology; PyTorch-profiler traces collected locally | Nsight not available on macOS; PyTorch profiler measured |
| Goodput vs. throughput (S1) | Deadline-aware goodput metric in the benchmark and serving layer | Measured |
| Whole-system view (S1) | Host-to-device transfer, queueing and service time measured separately | Measured |
| Hidden synchronizations (S3) | Harness design; sync-count tests; documented contamination sources | Measured |
| Train/serve parity, replay (S6) | Deterministic request-trace replay (Phase 4) | Measured |
| Distributed collectives, interconnects (S1) | **Out of scope.** Single-device project; stated in KNOWN_LIMITATIONS | n/a |
| PTX/SASS reading (S1) | Planned inspection of Triton-generated PTX on a real GPU | Pending GPU access |

## Uncertainties

- The exact GPU models and inference stack in production are not publicly verified.
- No NVIDIA GPU is available on the development machine (Apple M4). CUDA-specific claims stay labeled
  *pending* until they are measured on real hardware. See [GPU_VALIDATION_PLAN.md](GPU_VALIDATION_PLAN.md).
