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

## Phase 1: correctness baseline

```bash
.venv/bin/gpulab impls                                   # execution modes and support matrix
.venv/bin/gpulab run --device mps --dtype float16 --rows 8
.venv/bin/python scripts/calibrate_tolerances.py         # regenerates docs/CORRECTNESS.md numbers
.venv/bin/python -m pytest tests/test_correctness.py -v
```

Further sections (workload generation, benchmarks, experiments, dashboard) are added as each phase
lands.
