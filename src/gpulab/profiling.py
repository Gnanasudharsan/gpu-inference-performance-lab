"""PyTorch-profiler summaries: dispatches per forward, top operators, optional Chrome trace.

On CUDA the profiler records device kernels directly. On MPS and CPU the PyTorch profiler
has no device timeline, so the *number of leaf operator dispatches* (non-view ATen ops) is
reported as the proxy for kernel launches: on MPS each one encodes at least one Metal
compute (or blit) command. For NVIDIA timelines use Nsight Systems
(scripts/profile_nsys.sh). Nsight is not available on macOS.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from gpulab import impls
from gpulab.device import synchronize
from gpulab.workload import ModelConfig, make_inputs, make_weights

# ATen ops that only create views or metadata (no kernel launch).
VIEW_OPS = frozenset(
    "aten::" + n
    for n in (
        "view",
        "reshape",
        "_reshape_alias",
        "t",
        "transpose",
        "permute",
        "expand",
        "as_strided",
        "unsqueeze",
        "squeeze",
        "slice",
        "select",
        "detach",
        "alias",
        "_unsafe_view",
        "empty",
        "empty_strided",
        "resolve_conj",
        "resolve_neg",
        "split",
        "chunk",
        "narrow",
        "lift_fresh",
        "result_type",
        "unbind",
        "split_with_sizes",
        # conversions that launch nothing when they have no child copy (no-op casts)
        "to",
        "contiguous",
        "type_as",
    )
)


def _leaf_ops(prof: Any) -> list[Any]:
    leaves = []
    for ev in prof.events():
        if not ev.name.startswith("aten::") or ev.name in VIEW_OPS:
            continue
        has_aten_child = any(
            c.name.startswith("aten::") and c.name not in VIEW_OPS for c in ev.cpu_children
        )
        if not has_aten_child:
            leaves.append(ev)
    return leaves


def profile_forward(
    impl_name: str,
    model: ModelConfig,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
    iters: int = 20,
    trace_path: str | Path | None = None,
) -> dict[str, Any]:
    weights = make_weights(model, device, dtype)
    x = make_inputs(rows, model.d_model, device=device, dtype=dtype)
    runner = impls.get(impl_name).build(weights, rows=rows)
    for _ in range(10):
        runner(x)
    synchronize(device)
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as prof:
        for _ in range(iters):
            with record_function("gpulab.forward"):
                runner(x)
            synchronize(device)
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))

    leaves = _leaf_ops(prof)
    counts = Counter(ev.name for ev in leaves)
    self_cpu: dict[str, float] = defaultdict(float)
    for ev in leaves:
        self_cpu[ev.name] += ev.self_cpu_time_total
    top = sorted(counts, key=lambda n: -self_cpu[n])[:12]
    out: dict[str, Any] = {
        "impl": impl_name,
        "device": device.type,
        "dtype": str(dtype).removeprefix("torch."),
        "rows": rows,
        "iters": iters,
        "dispatches_per_forward": len(leaves) / iters,
        "top_ops": [
            {
                "name": n,
                "calls_per_forward": counts[n] / iters,
                "self_cpu_us_per_forward": self_cpu[n] / iters,
            }
            for n in top
        ],
    }
    if device.type == "cuda":
        kernels = [
            e
            for e in prof.events()
            if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA
        ]
        out["cuda_kernels_per_forward"] = len(kernels) / iters
        out["cuda_kernel_us_per_forward"] = sum(e.device_time_total for e in kernels) / iters
    return out
