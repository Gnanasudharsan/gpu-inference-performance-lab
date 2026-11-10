# GPU Inference Performance Lab

A reproducible lab for optimizing small-batch, latency-sensitive ML inference on GPUs: going from a correct
PyTorch baseline to measured, fused GPU execution, and reporting the tradeoffs between latency, throughput,
goodput, correctness and cost.

> **Status:** under active construction. See [docs/PHASE_STATUS.md](docs/PHASE_STATUS.md).
> No result appears in this repository unless it was measured on the hardware recorded next to it.

## Quick start

```bash
make install    # creates .venv with uv and installs gpulab[dev]
make check      # lint + typecheck + tests
.venv/bin/gpulab env
```

## Documentation

- [Research alignment](docs/research-alignment.md)
- [Environment report](docs/ENVIRONMENT.md)
- [Workload](docs/WORKLOAD.md) and [correctness methodology](docs/CORRECTNESS.md)
- [Kernel optimization and results](docs/KERNELS.md)
- [Decisions](docs/DECISIONS.md)
- [Benchmark methodology](docs/BENCHMARK_METHODOLOGY.md) and [profiling](docs/PROFILING.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Known limitations](docs/KNOWN_LIMITATIONS.md)

## License

MIT. See [LICENSE](LICENSE).
