# Known limitations

- **No NVIDIA GPU.** No NVIDIA hardware was available during development. CUDA, Triton-on-GPU,
  CUDA-graph and Nsight results are not measured until the [GPU validation plan](GPU_VALIDATION_PLAN.md)
  runs.
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
- **Triton and CUDA graphs are unmeasured.** These paths are correctness-tested only (fp32/fp16 under
  the interpreter). bf16 under the interpreter is blocked by a Triton interpreter bug
  (`tests/test_triton_bf16_diagnostic.py`).
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
