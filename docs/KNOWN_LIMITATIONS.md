# Known limitations

- **NVIDIA evidence is one T4 session on a shared free-tier host.** See
  [NVIDIA_T4.md](NVIDIA_T4.md).
  - *What it measures:* a launch-bound Turing GPU (sm_75, no bf16) behind a slow 4-vCPU virtualized
    host, in one session.
  - *What it cannot tell you:* behavior on data-center GPUs (L4, A100, H100), where launch overhead
    and occupancy limits differ.
  - *No between-session CIs:* two passing sessions agreed within about ±0.03×, but that is not a CI.
- **Research input.** The Deep Research report mentioned in the project brief was not supplied. The
  alignment uses public primary sources instead (see [research-alignment.md](research-alignment.md)).
- **Single device.** Distributed training and collectives (NCCL, InfiniBand), which the role posting
  mentions, are out of scope.
- **Unified memory.** On Apple silicon the CPU and GPU share memory, so "host-to-device transfer" is a
  copy within unified memory and is cheaper than PCIe. Transfer costs measured here do not transfer to
  discrete GPUs.
- **Noisy development GPU.** The M4 laptop GPU also drives the display, and its power management
  produces clustered latency tails and run-to-run p50 variation of up to 2.2×. Absolute latencies
  from this machine are indicative. Paired, interleaved speedups are the robust quantity.
- **No device-side MPS timeline.** The PyTorch profiler has no MPS device activity, and Instruments
  is not installed. Device time comes from MPS events, and launch counts from leaf dispatches.
- **The Triton kernel is slower than eager on the T4 (0.14–0.62×).**
  - *What is measured:* the time is inside its two kernels (Nsight Systems).
  - *What is not established:* the cause. Under-occupancy at small M is a hypothesis; Nsight Compute
    was not run.
  - *Also unexplained:* why Triton is much slower in latency mode than pipelined.
- **`reduce-overhead` regresses at d ≥ 1024 on the T4.** The diagnosis is that its CUDA-graph trees
  copy the weights on every replay, which the profiles are consistent with. The static-weights fix
  is not applied or measured.
- **The Triton Nsight capture includes autotuning.** Per-kernel averages in that capture are
  contaminated, so only minimums are used. A capture range that starts after warm-up would fix it.
- **bf16 Triton is unverified.** The CPU interpreter has a bf16 bug
  (`tests/test_triton_bf16_diagnostic.py`), and the T4 has no bf16.
- **Launch counts for `torch.compile` are incomplete.** Inductor-generated Metal kernels are invisible
  to the ATen-level profiler proxy.
- **Serving sweep points are single 3-second runs.** Point-to-point noise is visible. Repeated runs
  with CIs per point are a follow-up.
- **Queue bound not tuned.** The serving runs used `max_queue = 256`. Expired-request dropping kept
  queueing at ≤ 10 ms anyway, but a deadline-derived bound would shed earlier and waste less work.
- **No stall watchdog.** Rare single-batch GPU stalls of 114–392 ms were observed; nothing detects
  or hedges them.
- **Single-device, single-worker serving.** There is no multi-stream pipelining of H2D/compute/D2H,
  and no multi-GPU.
- **Root cause of the 3–10 ms MPS round-trip stalls is undetermined.** The display-frame hypothesis
  was tested and rejected.
- **The public live backend is a CPU.** Modal requires a payment method for GPU functions, and none
  was added. Live runs on the public site compare CPU implementations; a user-run backend on a GPU
  machine shows GPU results.
- **Docker image not built locally.** The Docker daemon was unavailable and disk was limited, so
  `deploy/Dockerfile` is unverified until built on a CUDA host.
- **First NVIDIA session (Kaggle T4, torch 2.11+cu130, commit `1e4076a`): the correctness gate
  failed, so no benchmarks ran** (`results/gpu/kaggle-t4-20261110-1510/`). 265 tests passed on the
  real T4, including the Triton kernels in fp16 and fp32. Four failed:
  - *One test assumed no CUDA device* (fixed).
  - *`cuda_graph_eager` changed an output it had already returned when the graph was replayed
    again* (fp32, fp16, bf16). **Root cause, found with an on-GPU diagnostic:** the CUDA-graph
    runner did not own what the graph reads.
    - *What the evidence showed* (`results/gpu/diag-t4-20261110-1514/`): the cuBLAS-workspace
      hypothesis was disproved. The eager graph was clean in isolation, while the Triton graph
      broke, because Triton's packed weight copy was owned only by the inner runner, which had been
      garbage-collected.
    - *The fix:* `GraphRunner` owns the inner runner, the weights and the static buffers.
    - *The new test:* `test_graph_runner_owns_what_it_replays` drops every outside reference and
      overwrites freed memory with NaN before replaying, so the bug fails deterministically.
    - *Status:* **verified on the T4** (`results/gpu/diag-t4-20261110-1523/`). All variants are clean
      with references kept or dropped, strided or contiguous, eager and Triton. The CUDA-only and
      Triton test files pass on the real GPU (25 passed; bf16 Triton skipped below compute
      capability 8.0).
