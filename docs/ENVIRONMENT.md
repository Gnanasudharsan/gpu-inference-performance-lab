# Environment report

Captured on 2026-11-10 with `gpulab env` and system tools. Every benchmark record embeds its own copy of
this information, so this page only summarizes the development machine.

## Development machine (all measured results so far)

| Item | Value |
|---|---|
| Machine | MacBook Pro, Apple M4 (10 CPU cores, 10-core GPU, Metal 4) |
| Memory | 16 GiB unified (CPU and GPU share it) |
| OS | macOS 27.2 (build 26B123), arm64 |
| Python | 3.12.16 (uv-managed virtualenv) |
| PyTorch | 2.15.0, MPS backend available, no CUDA build |
| Custom Metal kernels | `torch.mps.compile_shader` available and verified against a reference |
| `torch.compile` | Works on CPU and MPS (verified with a small fused function) |
| Triton | **Not installable on macOS** (no wheels). Validated in Linux CI with `TRITON_INTERPRET=1` |
| NVIDIA GPU / driver / CUDA toolkit | **None** (`nvidia-smi`, `nvcc` not found) |
| Nsight Systems / Nsight Compute | **Not available** (`nsys`, `ncu` not found; not supported on macOS hosts) |
| Docker | Installed, daemon not running (not needed) |
| Free disk | ~7 GB (constraint: avoid large toolchains and container images locally) |

## Accounts and permissions

| Item | Status |
|---|---|
| GitHub CLI | Authenticated as `Gnanasudharsan` (scopes: repo, workflow, gist, read:org) |
| Git identity | `Gnanasudharsan <gnana1403@gmail.com>` (global config, used unchanged) |
| Vercel CLI | Authenticated |
| GPU cloud (Modal, Runpod, Lambda, etc.) | **No credentials configured.** Any billable GPU work needs explicit authorization and a spending cap |

## Implications

1. All measured numbers in this repository come from the Apple M4 CPU or the Apple M4 GPU through MPS,
   unless a record says otherwise.
2. CUDA, Triton-on-GPU and CUDA-graph code paths are implemented and tested for **correctness** only
   (Triton interpreter), until an NVIDIA GPU run is authorized.
3. Apple-GPU timings use `torch.mps.Event` plus host wall-clock time after `torch.mps.synchronize()`.
   They are not comparable to NVIDIA numbers and are never presented as such.
