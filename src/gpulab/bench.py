"""Benchmark harness.

Three measurement modes, which answer different questions:

``latency``     One request at a time; the host synchronizes after each one. Reports host
                wall-clock (what a caller waits: launch + execution + sync) and device-event
                time (GPU execution span) per request. The gap is host-side overhead.
``e2e``         Like ``latency`` but the input starts on the host and the output returns to
                the host: H2D copy -> compute -> D2H copy, each stage timed with events.
``throughput``  ``burst`` requests issued back to back with a single sync at the end, so
                host launch overhead overlaps device execution. Reports time per request.

Controls against measurement bias: warm-up by count *and* time (allocator, kernel
caches, compilation, Metal pipeline state), Python GC disabled during timed regions,
pre-allocated events, multiple trials separated by idle gaps (repeatability), and
recording of power/thermal state before and after.
"""

from __future__ import annotations

import functools
import gc
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from gpulab import impls
from gpulab.correctness import compare
from gpulab.device import device_name, dtype_name, memory_allocated, synchronize
from gpulab.provenance import system_state
from gpulab.records import ExperimentRecord
from gpulab.stats import deadline_goodput, repeatability, summarize
from gpulab.workload import ModelConfig, describe, make_inputs, make_weights, reference_forward

MODES = ("latency", "e2e", "throughput")


@dataclass(frozen=True)
class BenchConfig:
    iters: int = 300  # timed requests per (mode, shape), split across trials
    trials: int = 3
    warmup_iters: int = 30
    #: Time-based warm-up. Apple and NVIDIA GPUs both ramp clocks under sustained load;
    #: 0.25 s was measured to be too short on Apple M4 (trial CV ~0.18), see methodology.
    warmup_s: float = 1.0
    trial_gap_s: float = 0.05
    #: Idle time inserted before each latency-mode request (spin-wait on the host) to model
    #: sparse arrivals. 0 = back-to-back synchronous requests.
    idle_gap_us: float = 0.0
    #: Iterations for the separate, event-instrumented pass (0 disables it).
    device_iters: int = 100
    burst: int = 50  # requests per burst in throughput mode
    bursts: int = 6
    deadline_us: float | None = None
    keep_samples: bool = True
    check_correctness: bool = True

    def __post_init__(self) -> None:
        if self.iters < self.trials or self.trials < 1:
            raise ValueError("need iters >= trials >= 1")
        if self.burst < 1 or self.bursts < 1:
            raise ValueError("burst and bursts must be >= 1")
        if self.idle_gap_us < 0 or self.device_iters < 0:
            raise ValueError("idle_gap_us and device_iters must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _spin(us: float) -> None:
    if us <= 0:
        return
    end = time.perf_counter_ns() + int(us * 1e3)
    while time.perf_counter_ns() < end:
        pass


class _EventPair:
    """Device-side start/stop timing. Returns None on CPU (no device timeline)."""

    a: Any
    b: Any

    def __init__(self, device: torch.device) -> None:
        self.kind = device.type
        if self.kind == "cuda":
            self.a = torch.cuda.Event(enable_timing=True)
            self.b = torch.cuda.Event(enable_timing=True)
        elif self.kind == "mps":
            self.a = torch.mps.Event(enable_timing=True)
            self.b = torch.mps.Event(enable_timing=True)

    def start(self) -> None:
        if self.kind != "cpu":
            self.a.record()

    def stop(self) -> None:
        if self.kind != "cpu":
            self.b.record()

    def elapsed_us(self) -> float | None:
        if self.kind == "cpu":
            return None
        return float(self.a.elapsed_time(self.b)) * 1e3


def _events(device: torch.device, n: int) -> list[_EventPair]:
    return [_EventPair(device) for _ in range(n)]


@contextmanager
def _quiet_gc() -> Iterator[None]:
    gc.collect()
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


@dataclass
class ModeResult:
    mode: str
    host_us: list[float] = field(default_factory=list)
    device_us: list[float] = field(default_factory=list)
    stages_us: dict[str, list[float]] = field(default_factory=dict)
    trial_medians_us: list[float] = field(default_factory=list)


def _warmup(fn: Callable[[], Any], device: torch.device, cfg: BenchConfig) -> None:
    t_end = time.perf_counter() + cfg.warmup_s
    i = 0
    while i < cfg.warmup_iters or time.perf_counter() < t_end:
        fn()
        synchronize(device)
        i += 1


def _split(n: int, k: int) -> list[int]:
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def measure_latency(
    runner: impls.Runner, x: torch.Tensor, device: torch.device, cfg: BenchConfig
) -> ModeResult:
    """Host wall-clock per synchronous request, then a separate event-instrumented pass.

    The two passes are kept apart because recording device events on every request was
    measured to perturb host latency on MPS (higher median and variance).
    """
    res = ModeResult("latency")
    _warmup(lambda: runner(x), device, cfg)
    for t, n in enumerate(_split(cfg.iters, cfg.trials)):
        if t:
            time.sleep(cfg.trial_gap_s)
        host = np.empty(n)
        with _quiet_gc():
            for i in range(n):
                _spin(cfg.idle_gap_us)
                t0 = time.perf_counter_ns()
                runner(x)
                synchronize(device)
                host[i] = (time.perf_counter_ns() - t0) / 1e3
        res.host_us.extend(host.tolist())
        res.trial_medians_us.append(float(np.median(host)))
    if device.type != "cpu" and cfg.device_iters:
        evs = _events(device, cfg.device_iters)
        with _quiet_gc():
            for ev in evs:
                _spin(cfg.idle_gap_us)
                ev.start()
                runner(x)
                ev.stop()
                synchronize(device)
        res.device_us = [float(e.elapsed_us() or 0.0) for e in evs]
    return res


def measure_e2e(
    runner: impls.Runner, x: torch.Tensor, device: torch.device, cfg: BenchConfig
) -> ModeResult:
    """Host tensor in, host tensor out. Stage breakdown comes from a separate instrumented pass."""
    res = ModeResult("e2e")
    nb = device.type == "cuda"
    x_host = x.detach().cpu()
    if nb:
        x_host = x_host.pin_memory()

    def once(ev: list[_EventPair] | None = None) -> torch.Tensor:
        if ev:
            ev[0].start()
        xd = x_host.to(device, non_blocking=nb)
        if ev:
            ev[0].stop()
            ev[1].start()
        y = runner(xd)
        if ev:
            ev[1].stop()
            ev[2].start()
        out = y.to("cpu", non_blocking=nb)
        if ev:
            ev[2].stop()
        return out

    _warmup(once, device, cfg)
    for t, n in enumerate(_split(cfg.iters, cfg.trials)):
        if t:
            time.sleep(cfg.trial_gap_s)
        host = np.empty(n)
        with _quiet_gc():
            for i in range(n):
                _spin(cfg.idle_gap_us)
                t0 = time.perf_counter_ns()
                once()
                synchronize(device)
                host[i] = (time.perf_counter_ns() - t0) / 1e3
        res.host_us.extend(host.tolist())
        res.trial_medians_us.append(float(np.median(host)))
    if device.type != "cpu" and cfg.device_iters:
        evs = [_events(device, 3) for _ in range(cfg.device_iters)]
        with _quiet_gc():
            for ev in evs:
                _spin(cfg.idle_gap_us)
                once(ev)
                synchronize(device)
        res.stages_us = {
            stage: [float(ev[k].elapsed_us() or 0.0) for ev in evs]
            for k, stage in enumerate(("h2d", "compute", "d2h"))
        }
    return res


def measure_throughput(
    runner: impls.Runner, x: torch.Tensor, device: torch.device, cfg: BenchConfig
) -> ModeResult:
    res = ModeResult("throughput")
    _warmup(lambda: runner(x), device, cfg)
    per_req = []
    with _quiet_gc():
        for _ in range(cfg.bursts):
            synchronize(device)
            t0 = time.perf_counter_ns()
            for _ in range(cfg.burst):
                runner(x)
            synchronize(device)
            per_req.append((time.perf_counter_ns() - t0) / 1e3 / cfg.burst)
    res.host_us = per_req
    res.trial_medians_us = per_req
    return res


_MEASURE = {"latency": measure_latency, "e2e": measure_e2e, "throughput": measure_throughput}


def _summary_block(samples: list[float]) -> dict[str, Any] | None:
    return summarize(samples).to_dict() if samples else None


@dataclass
class Prepared:
    impl: str
    runner: impls.Runner
    build_s: float
    first_call_us: float
    correctness: dict[str, Any] | None


def prepare(
    impl_name: str,
    weights: Any,
    x: torch.Tensor,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
    check: bool = True,
) -> Prepared:
    """Build a runner, time the build and the cold first call, and verify correctness."""
    impl = impls.get(impl_name)
    reason = impl.unsupported_reason(device, dtype)
    if reason:
        raise RuntimeError(f"{impl_name} unsupported on {device.type}/{dtype}: {reason}")
    t0 = time.perf_counter()
    runner = impl.build(weights, rows=rows)
    synchronize(device)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter_ns()
    y = runner(x)
    synchronize(device)
    first_call_us = (time.perf_counter_ns() - t0) / 1e3
    correctness = None
    if check:
        rep = compare(y, reference_forward(weights, x), dtype)
        correctness = rep.to_dict()
        if not rep.passed:
            raise AssertionError(f"{impl_name} failed correctness before benchmarking: {rep}")
    return Prepared(impl_name, runner, build_s, first_call_us, correctness)


def _record(
    prep: Prepared,
    res: ModeResult,
    model: ModelConfig,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
    cfg: BenchConfig,
    input_seed: int,
    mem_before: int | None,
    state_before: dict[str, Any],
    extra_params: dict[str, Any] | None = None,
) -> ExperimentRecord:
    host = _summary_block(res.host_us)
    metrics: dict[str, Any] = {
        "mode": res.mode,
        "host_us": host,
        "device_us": _summary_block(res.device_us),
        "stages_us": {k: _summary_block(v) for k, v in res.stages_us.items()} or None,
        "repeatability": repeatability(res.trial_medians_us),
        "trial_medians_us": res.trial_medians_us,
        "build_s": prep.build_s,
        "first_call_us": prep.first_call_us,
        "memory": {
            "allocated_before_bytes": mem_before,
            "allocated_after_bytes": memory_allocated(device),
            "driver_allocated_bytes": (
                int(torch.mps.driver_allocated_memory()) if device.type == "mps" else None
            ),
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        },
    }
    if host:
        metrics["throughput_rows_per_s"] = rows / (host["p50"] * 1e-6)
        if cfg.deadline_us:
            metrics["goodput"] = deadline_goodput(res.host_us, rows, cfg.deadline_us)
    if cfg.keep_samples:
        metrics["samples_host_us"] = [round(v, 3) for v in res.host_us]
        if res.device_us:
            metrics["samples_device_us"] = [round(v, 3) for v in res.device_us]
    return ExperimentRecord(
        kind="benchmark",
        impl=prep.impl,
        device=device.type,
        device_name=device_name(device),
        dtype=dtype_name(dtype),
        rows=rows,
        model=model.to_dict(),
        metrics=metrics,
        correctness=prep.correctness,
        params={
            "bench": cfg.to_dict(),
            "input_seed": input_seed,
            "torch_threads": torch.get_num_threads(),
            "workload": describe(model, rows, dtype),
            "system_state_before": state_before,
            "system_state_after": system_state(),
            **(extra_params or {}),
        },
    )


def _release(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def run_benchmark(
    impl_name: str,
    model: ModelConfig,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
    cfg: BenchConfig,
    modes: tuple[str, ...] = ("latency",),
    input_seed: int = 1234,
) -> list[ExperimentRecord]:
    """Benchmark one (impl, shape, device, dtype); one record per mode."""
    for m in modes:
        if m not in MODES:
            raise ValueError(f"unknown mode {m!r}")
    weights = make_weights(model, device, dtype)
    x = make_inputs(rows, model.d_model, input_seed, device, dtype)
    synchronize(device)
    mem_before = memory_allocated(device)
    prep = prepare(impl_name, weights, x, rows, device, dtype, cfg.check_correctness)
    state_before = system_state()
    records = [
        _record(
            prep,
            _MEASURE[m](prep.runner, x, device, cfg),
            model,
            rows,
            device,
            dtype,
            cfg,
            input_seed,
            mem_before,
            state_before,
        )
        for m in modes
    ]
    del prep, weights, x
    _release(device)
    return records


def run_interleaved(
    impl_names: list[str],
    model: ModelConfig,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
    cfg: BenchConfig,
    input_seed: int = 1234,
    mode: str = "latency",
) -> list[ExperimentRecord]:
    """Comparison with implementations interleaved trial by trial.

    ``mode="latency"``: each trial is ``iters / trials`` synchronous requests.
    ``mode="throughput"``: each trial is one pipelined burst of ``cfg.burst`` requests
    (one sample per trial), which is far less sensitive to power-state noise.

    Each trial measures every implementation once, rotating the order, so slow drift in
    clocks, thermals or background load affects all implementations equally. Per-trial
    medians are stored so speedups can be computed from *paired* trials.
    """
    if len(set(impl_names)) != len(impl_names) or not impl_names:
        raise ValueError("impl_names must be non-empty and unique")
    if mode not in ("latency", "throughput"):
        raise ValueError("interleaved mode must be 'latency' or 'throughput'")
    weights = make_weights(model, device, dtype)
    x = make_inputs(rows, model.d_model, input_seed, device, dtype)
    synchronize(device)
    mem_before = memory_allocated(device)
    preps = {
        n: prepare(n, weights, x, rows, device, dtype, cfg.check_correctness) for n in impl_names
    }
    for p in preps.values():
        _warmup(functools.partial(p.runner, x), device, cfg)
    state_before = system_state()
    results = {n: ModeResult(mode) for n in impl_names}
    short = BenchConfig(**{**cfg.to_dict(), "warmup_iters": 5, "warmup_s": 0.0})
    for t, n_iter in enumerate(_split(cfg.iters, cfg.trials)):
        k = t % len(impl_names)
        order = impl_names[k:] + impl_names[:k]
        for name in order:
            runner = preps[name].runner
            _warmup(functools.partial(runner, x), device, short)
            if mode == "throughput":
                with _quiet_gc():
                    synchronize(device)
                    t0 = time.perf_counter_ns()
                    for _ in range(cfg.burst):
                        runner(x)
                    synchronize(device)
                    per_req = (time.perf_counter_ns() - t0) / 1e3 / cfg.burst
                results[name].host_us.append(per_req)
                results[name].trial_medians_us.append(per_req)
                continue
            host = np.empty(n_iter)
            with _quiet_gc():
                for i in range(n_iter):
                    _spin(cfg.idle_gap_us)
                    t0 = time.perf_counter_ns()
                    runner(x)
                    synchronize(device)
                    host[i] = (time.perf_counter_ns() - t0) / 1e3
            results[name].host_us.extend(host.tolist())
            results[name].trial_medians_us.append(float(np.median(host)))
    if device.type != "cpu" and cfg.device_iters and mode == "latency":
        for name in impl_names:
            evs = _events(device, cfg.device_iters)
            runner = preps[name].runner
            with _quiet_gc():
                for ev in evs:
                    _spin(cfg.idle_gap_us)
                    ev.start()
                    runner(x)
                    ev.stop()
                    synchronize(device)
            results[name].device_us = [float(e.elapsed_us() or 0.0) for e in evs]
    group = f"{device.type}-{dtype_name(dtype)}-r{rows}-{time.time_ns():x}"
    records = [
        _record(
            preps[n],
            results[n],
            model,
            rows,
            device,
            dtype,
            cfg,
            input_seed,
            mem_before,
            state_before,
            {"interleaved": {"group": group, "peers": impl_names, "order": "rotate", "mode": mode}},
        )
        for n in impl_names
    ]
    del preps, weights, x
    _release(device)
    return records
