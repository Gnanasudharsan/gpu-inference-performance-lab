# Profiling methodology

## Tools by platform

| Tool | Platform | Status in this repo |
|---|---|---|
| PyTorch profiler (`gpulab profile`) | all | **Used.** Operator dispatches per forward, CPU-side dispatch cost, Chrome traces (`--trace`) |
| `torch.mps.Event` / `torch.cuda.Event` | MPS / CUDA | **Used** for device time (separate instrumented pass) |
| Nsight Systems (`scripts/profile_nsys.sh`) | NVIDIA | Written, **not executed** (no NVIDIA GPU; Nsight is unsupported on macOS hosts) |
| Nsight Compute (`scripts/profile_ncu.sh`) | NVIDIA | Written, **not executed** |
| Triton interpreter (`TRITON_INTERPRET=1`) | any | **Used** in CI for kernel correctness. It says nothing about performance |
| Xcode Instruments (Metal System Trace) | macOS | Not available: `xctrace` is missing because only the Command Line Tools are installed |

On MPS, the PyTorch profiler has no device timeline. Each *leaf* ATen dispatch (an op with no
non-view ATen children, excluding view and no-op conversions) therefore stands in for a launch: it
encodes at least one Metal compute or blit command.

## Baseline profile (measured, Apple M4, eager, fp16, 8 rows, 1024×2816×4)

From `results/phase2/profile_eager_mps.json`:

| Op | Calls/forward | Note |
|---|---|---|
| `aten::linear` | 12 | 3 GEMMs × 4 blocks: the only ops that do the "real" work |
| `aten::copy_` | 16 | dtype casts around the fp32 RMSNorm (upcast x, upcast g, downcast) |
| `aten::mul` | 12 | norm scale, norm weight, gate product |
| `aten::add` | 8 | `+ eps` and the residual |
| `aten::pow`, `mean`, `rsqrt`, `silu` | 4 each | rest of the norm, activation |
| **total** | **64** | 16 per block, of which only 3 are GEMMs |

The CPU profile (fp32, `results/phase2/profile_eager_cpu.json`) is dominated by `aten::mm`, about
8.9 ms of 64 dispatches. On CPU the GEMMs themselves are the bottleneck. On the GPU, the other 13
dispatches per block plus their memory round-trips are the overhead to remove.

## What the profile predicts

1. Fusing the norm into the GEMM removes 9–10 dispatches per block and two activation round-trips.
2. Packing gate and up removes one GEMM launch and lets SiLU·mul run in the epilogue.
3. Folding the residual into the down-projection epilogue removes the last add.
4. Floor: 2 dispatches per block (8 per forward), each reading its weights exactly once.

Phase 3 implements this and checks it against measurement.

## How to profile

```bash
gpulab profile --device mps --dtype float16 --rows 8 --impl eager --trace /tmp/eager.json
# on an NVIDIA host:
scripts/profile_nsys.sh eager 8 && scripts/profile_nsys.sh triton 8
scripts/profile_ncu.sh 8
```
