# Correctness methodology

Every implementation is compared against the float64 CPU reference (`reference_forward`). Errors are
scale-normalized:

- `rel_l2 = ‖y − ref‖₂ / ‖ref‖₂`
- `max_scaled = max|y − ref| / max|ref|`

A result passes when it is finite and both metrics are within the dtype's tolerance.

## Tolerance calibration (measured)

`scripts/calibrate_tolerances.py` runs the eager baseline over 20 shapes per dtype (`d_model` 256–2048,
depth 1–8, rows 1–64). The worst case observed on 2026-11-10 (Apple M4, PyTorch 2.15.0):

| Device / dtype | worst `rel_l2` | worst `max_scaled` | tolerance `rel_l2` | tolerance `max_scaled` |
|---|---|---|---|---|
| CPU fp32 | 1.02e-6 | 1.54e-6 | 1e-5 | 1e-4 |
| M4 GPU (MPS) fp32 | 1.02e-6 | 1.50e-6 | 1e-5 | 1e-4 |
| CPU fp16 | 7.26e-4 | 1.28e-3 | 3e-3 | 1e-2 |
| M4 GPU (MPS) fp16 | 7.26e-4 | 1.29e-3 | 3e-3 | 1e-2 |
| CPU bf16 | 6.14e-3 | 1.06e-2 | 2e-2 | 8e-2 |
| M4 GPU (MPS) bf16 | 6.13e-3 | 1.06e-2 | 2e-2 | 8e-2 |

The tolerances sit 3–10× above the baseline's own error. That leaves room for legitimate reordering
(fused epilogues round once instead of twice; split-K changes the summation order) while staying far
below the O(1) errors that real bugs produce.

## The checker is tested too

`tests/test_correctness.py::test_checker_detects_bugs` injects four realistic bugs and asserts that each
one **fails** in every dtype:

- gate and up weights swapped
- residual dropped
- wrong ε
- rows permuted

Further tests check NaN detection and shape mismatches.

## Coverage of the eager baseline

`tests/test_correctness.py` covers:

- every available device (CPU, MPS here; CUDA when present) and every dtype (fp32, fp16, bf16)
- edge shapes: a single row, non-power-of-two widths (96, 200), odd row counts (3, 17), and 64 rows
- non-contiguous strided inputs and inputs with extra leading dimensions
- repeated execution, which must be bitwise identical and must not mutate the input
- input validation (wrong width raises an error)

## Baseline records

`results/phase1/correctness.jsonl` holds 18 records: eager × {CPU, M4 GPU} × {fp32, fp16, bf16} ×
rows {1, 8, 64} at the default 1024×2816×4 shape. Each record has full provenance. All pass.
