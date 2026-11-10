"""Serving experiments: replay a trace through a server and summarize the outcome."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from gpulab.device import device_name, dtype_name
from gpulab.records import ExperimentRecord
from gpulab.serving.server import ReplayResult, Server, ServerConfig
from gpulab.serving.trace import Request, TraceSpec, make_trace
from gpulab.stats import summarize
from gpulab.workload import ModelConfig, make_inputs, make_weights


def make_request_inputs(
    trace: list[Request], d_model: int, dtype: torch.dtype, seed: int = 10_000
) -> list[torch.Tensor]:
    """Deterministic host-side payload per request (generated before replay, not timed)."""
    return [make_inputs(r.rows, d_model, seed + r.rid, "cpu", dtype) for r in trace]


def summarize_replay(res: ReplayResult, duration_s: float) -> dict[str, Any]:
    outs = res.outcomes
    n = len(outs)
    ok = [o for o in outs if o.status == "ok"]
    on_time = [o for o in ok if o.on_time]
    wall_s = max(res.wall_us, 1.0) * 1e-6
    counts = {s: sum(o.status == s for o in outs) for s in ("ok", "rejected", "expired", "error")}

    def summ(vals: list[float]) -> dict[str, Any] | None:
        return summarize(vals).to_dict() if vals else None

    batch_rows = [o.batch_rows for o in ok if o.batch_rows]
    padded = [o.padded_rows for o in ok if o.padded_rows]
    util = res.busy_us / max(res.wall_us, 1.0)
    return {
        "requests": n,
        "counts": counts,
        "offered_rps": n / duration_s,
        "completed_rps": len(ok) / wall_s,
        "throughput_rows_per_s": sum(o.rows for o in ok) / wall_s,
        "goodput_rows_per_s": sum(o.rows for o in on_time) / wall_s,
        # every request that does not finish on time counts as a miss (incl. shed/expired)
        "deadline_miss_rate": 1.0 - len(on_time) / n if n else 0.0,
        "latency_us": summ([o.latency_us for o in ok if o.latency_us is not None]),
        "queue_us": summ([o.queue_us for o in ok if o.queue_us is not None]),
        "service_us": summ([o.service_us for o in ok if o.service_us is not None]),
        "mean_batch_rows": float(np.mean(batch_rows)) if batch_rows else None,
        "padding_overhead": (float(np.sum(padded)) / float(np.sum(batch_rows)) - 1.0)
        if batch_rows
        else None,
        "batches": res.batches,
        "utilization": util,
        "saturated": util > 0.95 or counts["rejected"] > 0,
        "peak_memory_bytes": res.peak_memory_bytes,
        "calibration_us": {str(k): v for k, v in res.calibration_us.items()},
        "errors": res.errors[:5],
        # replay fidelity: how late idle wake-ups were (should be << deadlines)
        "wake_lateness_us": summ(res.wake_lateness_us),
    }


def run_serving_experiment(
    impl_name: str,
    model: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
    server_cfg: ServerConfig,
    trace_spec: TraceSpec,
    server: Server | None = None,
) -> tuple[ExperimentRecord, ReplayResult]:
    weights = None
    if server is None:
        weights = make_weights(model, device, dtype)
        server = Server(impl_name, weights, server_cfg)
    else:
        server.cfg = server_cfg
    trace = make_trace(trace_spec)
    inputs = make_request_inputs(trace, model.d_model, dtype)
    res = server.replay(trace, inputs)
    metrics = summarize_replay(res, trace_spec.duration_s)
    rec = ExperimentRecord(
        kind="serving",
        impl=impl_name,
        device=device.type,
        device_name=device_name(device),
        dtype=dtype_name(dtype),
        rows=max(trace_spec.rows_choices),
        model=model.to_dict(),
        metrics=metrics,
        params={"server": server_cfg.to_dict(), "trace": asdict(trace_spec)},
    )
    return rec, res
