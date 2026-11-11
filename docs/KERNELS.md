# Kernel optimization

## Starting point: measured bottlenecks

The eager baseline (Apple M4 GPU, fp16, 8 rows, 1024×2816×4) issues **64 leaf dispatches per forward**,
16 per block, and only 3 of those are GEMMs ([PROFILING.md](PROFILING.md)). At 8 rows the forward reads
69 MB of weights at ~1 FLOP/byte, so its floor is *weight bytes ÷ bandwidth*: about 0.58 ms at the
M4's ~120 GB/s nominal. Eager in pipelined mode took ~1.1–1.2 ms (Phase 2). The remaining gap is
dispatch overhead plus intermediate round-trips.

## Escalation ladder

| Step | Implementation | Dispatches / block | What changes |
|---|---|---|---|
| 0 | `eager` | 16 | baseline |
| 1 | `fused_weights` | measured below | packed gate‖up GEMM, native `F.rms_norm`, residual folded into `addmm` |
| 2 | `compile` | measured below | Inductor fuses the elementwise work. GEMMs stay vendor GEMMs |
| 3a | `metal` (Apple) | **2** | fused norm + gate/up GEMV + SiLU·mul; fused down GEMV + residual |
| 3b | `triton` (NVIDIA) | **2** | same fusion structure as tiled `tl.dot` GEMMs |
| 4 | `cuda_graph_*`, `compile_cudagraphs` (NVIDIA) | 1 graph launch / forward | removes per-kernel CPU launch cost |

### RMSNorm folded in by linearity

`RMSNorm(x)·W = (s ⊙ x ⊙ g)·W = s ⊙ ((x ⊙ g)·W)` with `s = rsqrt(mean(x²) + ε)` per row. The fused
kernels compute `(x ⊙ g)·W` and apply `s` in the epilogue. In Triton, `Σx²` accumulates *inside the same
K-loop* that streams `x` through `tl.dot`, so the norm costs no extra pass over memory. In Metal, each
threadgroup recomputes `s` for its rows; `x` is a few KB and cache-resident. Rounding differs from
eager, which rounds `n` to the working dtype once more. The difference stays within tolerance and is
covered by the tests.

## Metal GEMV kernels (Apple M4): design and iteration

Shape of the problem: `rows ≤ 8`, so each GEMM is a skinny matrix product (effectively 1–8 GEMVs
sharing one weight read). The bandwidth-optimal pattern:

- One simdgroup (32 lanes) owns `COLS` output columns, which are weight rows (contiguous in memory).
- Lanes stride the reduction dimension with 4-wide vector loads, so each weight row is read with
  coalesced, contiguous accesses **exactly once per forward**.
- Accumulation is fp32 in registers, then `simd_sum`. There is no threadgroup memory and no barrier
  in the hot loop.
- The epilogue fuses SiLU(gate)·up, or the residual add, and writes the working dtype.

**v1 → v2** (both kept in history: `cb577af` and `8cff002`):

| | v1 | v2 |
|---|---|---|
| Row count | runtime loop bound (`R` buffer) | specialized per R = 1…8 at compile time |
| Column loop | outside the K-loop: `x` re-read for every column | inside the K-loop: each `x` load reused across `COLS` columns |

v1 was competitive at 1 row and **regressed at 8 rows**. Per output column it re-read every row of `x`,
about 4× more `x` loads than weight loads at R = 8. Its runtime-bounded accumulator arrays also could
not be fully register-allocated. The interleaved A/B in `results/phase3/metal_v1_vs_v2.json` gives
the numbers.

### Launch configuration

`scripts/tune_metal.py` measures 12 configurations (simdgroups/threadgroup ∈ {1,2,4,8} × columns/simdgroup
∈ {1,2,4}) interleaved in pipelined mode. Result (`results/phase3/metal_tuning.json`):

- 8 simdgroups × 2 columns for rows ≤ 4 (within 0.4% of best at 1 and 2 rows; best at 4)
- 8 × 1 for rows 5–8 (best at 8 rows)
- 4 columns at 8 rows was 27–30% slower. *Inference:* register pressure from 8×4×2 fp32 accumulators per lane.

### Above 8 rows

The GEMV formulation stops being the right tool once the product has enough rows for compute to
matter. At rows > 8 the `metal` path uses MPS's GEMM (simdgroup-matrix hardware) for the products,
and custom kernels only for RMSNorm and SiLU·mul: 4 dispatches per block instead of 2. Whether this
beats eager at large batch is measured below, not assumed.

## Triton kernels (NVIDIA)

`src/gpulab/kernels/triton_kernels.py`:

- **`norm_gate_up_silu`**: a 2-D grid over (row tiles, hidden tiles). The K-loop loads an `x` tile
  once, multiplies by `g`, casts to the working dtype, and issues two `tl.dot`s against the matching
  gate and up weight tiles (two fp32 accumulators). The epilogue applies `rsqrt` and SiLU·mul. The
  `(rows, 2H)` intermediate never exists.
- **`down_residual`**: tiled GEMM with the residual add in the epilogue.
- fp32 inputs use `input_precision="ieee"` (no silent TF32).
- Autotuning covers BLOCK_M ∈ {16, 32, 64} (16 is `tl.dot`'s minimum), BLOCK_N ∈ {32, 64, 128},
  BLOCK_K ∈ {64, 128}, and warps/stages. The interpreter uses one small config.

**Verified:**
- fp32 and fp16 correctness on all test shapes, under `TRITON_INTERPRET=1` in CI and on a real T4
  (the session's 289-test correctness gate).

**Measured on the T4** ([NVIDIA_T4.md](NVIDIA_T4.md)):
- **Slower than eager everywhere:** 0.38× at 8 rows on the default model, 0.14–0.62× across the
  sweep.
- **Launches are not the cause.** It issues 8 kernels per forward against eager's 60, and CUDA-graph
  replay leaves it unchanged.
- **Nsight Systems puts the time inside the kernels.** The fastest instance per layer is 340 µs for
  gate/up and 241 µs for down, about 24–34 GB/s effective. cuBLAS GEMMs reach about 168 GB/s.
- **The autotune key is now bucketed to powers of two** (`gpulab.kernels.tuning.autotune_m_key`),
  so serving warm-up covers every key. Keying on exact M re-tuned during serving and caused a 12.7 s
  p99.

**Not verified:**
- *bf16:* a bare bf16 `tl.dot` is wrong under the Triton 3.9 CPU interpreter
  (`tests/test_triton_bf16_diagnostic.py`, strict xfail), and the T4 (sm_75) has no bf16.
- *register and shared-memory limits:* Nsight Compute was not run.

**Known design limit.** At M ≤ 16 the grid is only `ceil(N / BLOCK_N)` programs: 22–88 for the down
projection here, fewer than the SM count of large GPUs. Split-K would raise occupancy. It needs a
two-phase reduction because the SiLU and residual epilogues are applied after the full sum. On the
40-SM T4 this is the leading hypothesis for the measured slowdown; it is not yet confirmed.

## CUDA Graphs

`cuda_graph_eager` and `cuda_graph_triton` capture the whole forward once per row count and replay
it with one launch. The runner copies the request into a static input buffer and clones the static
output; both copies are included in measurements. A request with a different row count is rejected.
The serving layer buckets and pads instead, so this never happens there. `compile_cudagraphs` uses
Inductor's `reduce-overhead` mode. All three are implemented and tested for failure modes.

**Measured on the T4** (fp16, 1024×2816×4, 8 rows, pipelined):
- `cuda_graph_eager` is **1.98× [1.96, 1.99]** over eager, the largest gain on that host. It
  shrinks with model size: 2.35–2.59× at d=512, 1.00–1.13× at d=2048.
- `cuda_graph_triton` is 0.40×; graphs cannot rescue a GPU-bound kernel.
- `compile_cudagraphs` is 0.99× here, 2.99× at d=512 and 0.36–0.41× at d=2048. The likely cause is
  per-replay weight copies (see [NVIDIA_T4.md](NVIDIA_T4.md)).
- **A bug the T4 found:** the graph runner did not own the memory its graph reads, so a replay after
  garbage collection read freed memory. `GraphRunner` now owns the inner runner, the weights and
  the buffers, and `test_graph_runner_owns_what_it_replays` pins it.

## Measured results (Apple M4 GPU via MPS, PyTorch 2.15.0, commit `758dbd6`)

All comparisons are interleaved: the implementation order rotates every trial, and there are 20
trials. Speedup is the geometric mean of paired per-trial ratios (eager ÷ candidate), with a 95%
bootstrap CI. **Verdicts:** *faster* means the CI lies entirely above 1; *slower* means it lies
entirely below 1; anything else is reported as *no significant difference*. Raw data is in
`results/phase3/bench_*.jsonl`, and the full table is in `results/phase3/REPORT.md` and the dashboard.

### Launches per forward (1024×2816×4)

| Implementation | ≤ 8 rows | > 8 rows |
|---|---|---|
| eager | 64 | 64 |
| fused_weights | 20 | 20 |
| metal | **8** (all custom) | 16 (8 custom + 8 MPS GEMM) |
| compile | ATen-visible: 8. Inductor kernels not counted (`launch_count_complete: false`) | same |

### Pipelined time per forward, fp16, 1024×2816×4 (most stable mode)

| rows | eager µs | fused_weights | compile | **metal** |
|---|---|---|---|---|
| 1 | 879 | 1.19× [1.15, 1.26] | 1.14× [1.07, 1.22] | 1.16× [1.11, 1.23] |
| 2 | 927 | 0.98× (n.s.) | 0.96× (n.s.) | **1.21× [1.16, 1.26]** |
| 4 | 989 | 1.04× [1.01, 1.08] | 0.96× (n.s.) | **1.26× [1.22, 1.30]** |
| 8 | 1,053 | 1.07× [1.03, 1.11] | 1.02× (n.s.) | **1.20× [1.14, 1.27]** |
| 16 | 1,220 | 1.08× [1.06, 1.10] | 1.02× (n.s.) | 1.00× (n.s.): GEMM fallback |
| 32 | 1,475 | 1.08× [1.05, 1.12] | 1.08× [1.03, 1.12] | 1.07× [1.04, 1.10] |
| 64 | 2,048 | 1.14× [1.10, 1.18] | 1.13× [1.08, 1.17] | 1.10× [1.07, 1.13] |

**Achieved weight-streaming bandwidth.** Figures are *derived* from the weight bytes and the time
above; the 120 GB/s reference is *theoretical* (vendor-nominal for the M4):

| rows | eager | fused_weights | compile | metal |
|---|---|---|---|---|
| 1 | 78.7 GB/s (66%) | 91.0 (76%) | 89.0 (74%) | 89.0 (74%) |
| 8 | 65.8 (55%) | 70.8 (59%) | 65.9 (55%) | **81.9 (68%)** |
| 64 | 33.8 (28%) | 38.6 (32%) | 37.6 (31%) | 36.8 (31%) |

### Across shapes and precisions (pipelined, metal vs eager)

| case | rows 1 | rows 8 |
|---|---|---|
| fp16, 512×1408×4 (17 MB) | **1.66× [1.52, 1.81]** | **1.62× [1.46, 1.80]** |
| fp16, 2048×5632×4 (277 MB) | 1.01× (n.s.) | 1.20× [1.15, 1.24] |
| fp32, 1024×2816×4 | 1.03× [1.02, 1.04] | 1.36× [1.27, 1.45] |
| bf16, 1024×2816×4 | 1.14× [1.09, 1.19] | 1.31× [1.22, 1.40] |

### Synchronous latency mode (one request at a time; noisier)

These comparisons are against eager, fp16, 1024×2816×4.

| rows | metal | fused_weights | compile |
|---|---|---|---|
| 1 | 1.05× (n.s.) | 1.18× [1.07, 1.32] | 0.98× (n.s.) |
| 2 | **1.33× [1.21, 1.49]** | 1.00× (n.s.) | 0.93× (n.s.) |
| 8 | **1.36× [1.11, 1.69]** | 1.10× (n.s.) | 1.04× (n.s.) |
| 16–64 | n.s. | n.s. | n.s. |

### Regressions (CI entirely below 1)

| case | speedup | interpretation |
|---|---|---|
| CPU fp32, `compile`, 1 row, latency | 0.78× | Inductor's CPU code for skinny GEMMs loses to the MKL/Accelerate-backed eager `mm` |
| CPU fp32, `compile`, 8 rows, latency / pipelined | 0.63× / 0.70× | same |
| CPU fp32, `fused_weights`, 8 rows, latency | 0.79× | the packed (2H×D) GEMM is slower than two separate GEMMs on this CPU (inference; not profiled further) |
| MPS fp32, `metal`, 32 rows, latency | 0.81× | in pipelined mode the same configuration is **1.09× faster**. The two modes disagree, so it is not a robust regression, but it is reported |

### Interpretation

1. **Where the kernels help.** The fused Metal GEMVs are fastest exactly where they were designed to
   be: 2–8 rows, where they cut 64 launches to 8 and lift streaming bandwidth from 55% to 68% of
   nominal. The gain is largest for the small model (1.6×), where fixed per-launch costs are the
   biggest share of the time.
2. **Where they stop helping.** Large models at 1 row (277 MB of weights) are already close to
   bandwidth-bound in eager. At ≥ 16 rows the GEMV formulation is replaced by MPS GEMM, so the
   remaining gain comes from epilogue fusion only (1.07–1.10×), or there is none (16 rows).
3. **`torch.compile`** is roughly neutral on MPS for this workload: the GEMMs dominate, and those
   stay vendor GEMMs. It is a regression on CPU. It is not used where it does not help.
4. **What is still on the table (not done).**
   - Compute `inv_rms` once per forward instead of per threadgroup.
   - Use simdgroup-matrix instructions for 9–32 rows instead of falling back to MPS GEMM.
   - Fuse across blocks into a persistent kernel.
   - Evaluate int8/fp8 weights: weight bytes are the floor, so halving them is the largest remaining
     lever, but it changes numerics and needs its own accuracy study.
