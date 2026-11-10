# Workload: small-batch gated-MLP inference

## Definition

A network is `depth` residual blocks of width `d_model`. Each block computes:

```
n = RMSNorm(x) = x · rsqrt(mean(x²) + ε) · g        (per row, statistics in fp32)
h = SiLU(n · W_gateᵀ) ⊙ (n · W_upᵀ)                 (d_hidden wide)
x ← x + h · W_downᵀ                                  (residual)
```

The default shape is `d_model = 1024`, `d_hidden = 2816` (≈2.75×), `depth = 4`. That is 34.6 M
parameters, or 69 MB in fp16. Requests have 1–64 rows.

Code: [`src/gpulab/workload.py`](../src/gpulab/workload.py). The golden reference is `reference_forward`,
which runs in float64 on the CPU and is itself validated against an independent NumPy implementation in
`tests/test_workload.py`.

## Why this is a meaningful GPU performance problem

1. **It is memory-bound at small batch.** A forward pass reads every weight once. With `rows = 1` the
   arithmetic intensity is about 1 FLOP/byte. At 64 rows it is about 60. GPUs need tens to hundreds of
   FLOP/byte to become compute-bound, so the latency floor at small batch is *weight bytes ÷ memory
   bandwidth*, not FLOPs. `gpulab.workload.describe` computes these figures.
2. **Launch overhead is comparable to useful work.** Eager PyTorch issues about 10 kernels per block
   (upcast, square, mean, rsqrt, two multiplies, downcast, three GEMMs, SiLU, gate multiply, residual
   add). Many of them touch only a few kilobytes at small batch, so each one's fixed launch cost (a few
   µs on CUDA, similar or more on Metal) dominates its actual work. This is the regime where CUDA graphs
   and fusion pay off.
3. **Intermediate round-trips are avoidable.** Writing `gate` and `up` (2 × `rows × d_hidden`) to
   memory and reading them back for SiLU and the product is pure overhead. A fused GEMM epilogue
   eliminates it. The same applies to the norm's intermediates and the residual add.
4. **It transfers to other models.** RMSNorm + SwiGLU is the feed-forward block of most modern
   transformers and of many tabular and sequence models, so the optimizations generalize.

## Determinism

Weights (seed `ModelConfig.seed`) and inputs (seed `input_seed`) are drawn on the CPU in float32 and only
then cast and moved, so every device and dtype sees identical values. A test checks this, and a second
test checks that repeated execution on a device is bitwise stable.

## Numerical contract (reduced precision)

| Stage | Storage | Compute |
|---|---|---|
| Activations, weights | working dtype | |
| RMSNorm statistics and scale | | fp32, rounded once |
| GEMMs | working dtype in/out | fp32 accumulate |
| SiLU and gate product | | fp32 internally. Eager rounds after SiLU and after the product; fused kernels round once |
| Residual add | working dtype | |

On NVIDIA GPUs, fp32 GEMMs must not silently use TF32. Triton kernels pass `input_precision="ieee"`
for fp32.
