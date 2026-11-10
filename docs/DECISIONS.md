# Engineering decisions

Each entry records what was decided, why, the alternatives that were rejected, and what would change
the decision. Newest entries go at the bottom.

## D1. Workload: a stack of pre-norm RMSNorm → SwiGLU gated-MLP → residual blocks

**Decision.** Each block computes `y = x + W_down( silu(W_gate · n) ⊙ (W_up · n) )`, where
`n = RMSNorm(x)`. A network is `depth` such blocks. Batch sizes are 1–64 rows; hidden widths are
256–2048 with a 2.67–4× expansion.

**Why.** At small batch, the GEMMs have arithmetic intensity of roughly `batch` FLOP/byte of weight
traffic, so they are memory-bandwidth-bound. Eager execution issues about 8–10 kernels per block
(the norm's reduction and scale, three GEMMs, SiLU, multiply, add). That makes launch overhead and
intermediate memory round-trips first-order costs, the same regime the public sources describe for
low-latency inference with models smaller than LLMs. The block also appears in most modern transformer
and MLP stacks, so the optimizations transfer.

**Rejected.** A full transformer (attention adds KV-cache and sequence-length dimensions that dilute
the low-latency story and are already covered by mature libraries). A plain ReLU MLP (too little
fusable structure). CNNs (not representative of tabular or sequential inference at small batch).

## D2. Development hardware is Apple M4; NVIDIA results are a separate, explicitly gated step

**Decision.** Measure everything possible on the Apple M4 CPU and GPU. Implement CUDA-only paths
(Triton kernels, CUDA graphs) and validate their correctness with the Triton CPU interpreter in Linux
CI. Mark their performance as *pending* until an authorized NVIDIA run happens.

**Why.** It is honest and reproducible on available hardware. The Apple GPU is a real GPU with real
launch overhead, memory bandwidth limits and a custom-kernel path (Metal via
`torch.mps.compile_shader`), so the methodology is exercised end to end.

**Rejected.** Simulating or estimating CUDA numbers (that would be fabrication). Renting a GPU without
authorization (the operating principles forbid it).

## D3. Python tooling: uv, hatchling, ruff, mypy, pytest

Fast, standard, and small on disk, which matters because only about 7 GB is free. No Poetry or conda.

## D4. CI runs on CPU-only PyTorch, plus a separate Triton-interpreter job

The Triton interpreter (`TRITON_INTERPRET=1`) runs `@triton.jit` kernels on CPU tensors with NumPy
semantics. It checks indexing, masking and numerics, but not performance, and it does not catch every
GPU-only failure (for example, shared-memory limits or register pressure). Those are covered by the GPU
validation plan.
