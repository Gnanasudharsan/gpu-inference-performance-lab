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
