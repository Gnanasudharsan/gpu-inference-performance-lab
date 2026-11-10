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

## Phase 2: benchmarks and profiles

See [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md#reproduce) for the exact commands that
produced `results/phase2/`, and [PROFILING.md](PROFILING.md) for profiler usage.

## Phase 3: kernels

```bash
python scripts/tune_metal.py --trials 20 --rows 1 2 4 8      # -> results/phase3/metal_tuning.json
python scripts/ab_metal_versions.py --trials 30              # v1 (cb577af) vs current, interleaved
scripts/phase3_benchmarks.sh                                 # full matrix -> results/phase3/
TRITON_INTERPRET=1 pytest -m triton                          # Linux, needs triton
```

Further sections (workload generation, benchmarks, experiments, dashboard) are added as each phase
lands.
