# Reproducibility

## Setup

```bash
git clone https://github.com/Gnanasudharsan/gpu-inference-performance-lab
cd gpu-inference-performance-lab
make install          # uv venv + editable install with dev extras
make check            # lint, types, tests
.venv/bin/gpulab env  # provenance report for your machine
```

Python 3.12 and PyTorch 2.4 or newer are required. The development machine used PyTorch 2.15.0.

Further sections (workload generation, benchmarks, experiments, dashboard) are added as each phase
lands.
