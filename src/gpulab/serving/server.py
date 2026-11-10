"""Single-device inference server with bounded queueing and deadline-aware batching.

Execution model: one device, one worker, non-preemptive service, open-loop arrivals.
Replay is a single-threaded discrete-event loop. Before every scheduling decision, all
requests whose arrival time has passed are admitted with their *true* arrival timestamps.
Because service is non-preemptive on one device, this yields exact queueing delays and
exact queue-occupancy at each arrival (used for admission control), without a client
thread competing for the GIL.

Policies
  direct   serve the oldest request alone, immediately.
  static   wait until ``batch_rows`` rows are queued (or ``static_timeout_us`` passes
           since the oldest arrival, so a quiet tail cannot starve), then serve them.
  dynamic  serve whatever is queued, up to ``max_batch_rows``. If the batch is not full,
           wait for more arrivals for at most ``max_wait_us``, and never beyond the point
           where the oldest request's deadline minus the predicted service time would be
           violated. Predictions come from a service-time table calibrated at startup.

Overload behaviour: the queue is bounded (``max_queue`` requests). Arrivals beyond it are
rejected immediately (load shedding). Optionally, requests that have already missed their
deadline when they reach the head are dropped instead of wasting device time on them.
A failing batch marks only its own requests as errors; the server keeps running.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from gpulab import impls
from gpulab.device import memory_allocated, synchronize
from gpulab.serving.trace import Request
from gpulab.workload import ModelWeights

POLICIES = ("direct", "static", "dynamic")
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class ServerConfig:
    policy: str = "dynamic"
    max_batch_rows: int = 32
    batch_rows: int = 8  # static policy
    max_wait_us: float = 500.0  # dynamic policy
    static_timeout_us: float = 20_000.0
    max_queue: int = 256
    drop_expired: bool = True
    calibration_iters: int = 30
    #: Safety margin kept between the predicted completion and the deadline when the dynamic
    #: policy waits. Without it, host wake-up jitter (macOS sleep overshoots by 100s of µs)
    #: turns "wait until the last moment" into a deadline miss (observed in testing).
    slack_margin_us: float = 300.0
    #: How the replay waits while idle. "spin" busy-waits (exact wake-ups; one CPU core busy,
    #: like a poll-mode server). "sleep" uses chunked sleeps: energy-friendly, but measured on
    #: macOS to wake up to ~10 ms late under background QoS. Lateness is always reported.
    idle_wait: str = "spin"

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if self.max_batch_rows < 1 or self.batch_rows < 1 or self.max_queue < 1:
            raise ValueError("batch sizes and queue bound must be >= 1")
        if self.batch_rows > self.max_batch_rows:
            raise ValueError("batch_rows must be <= max_batch_rows")
        if self.max_wait_us < 0 or self.static_timeout_us <= 0 or self.slack_margin_us < 0:
            raise ValueError("wait bounds must be non-negative")
        if self.idle_wait not in ("spin", "sleep"):
            raise ValueError("idle_wait must be 'spin' or 'sleep'")
        if self.max_batch_rows > BUCKETS[-1]:
            raise ValueError(f"max_batch_rows must be <= {BUCKETS[-1]}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Outcome:
    rid: int
    rows: int
    arrival_us: float
    deadline_us: float
    status: str = "pending"  # ok | rejected | expired | error
    start_us: float | None = None
    end_us: float | None = None
    batch_rows: int | None = None
    padded_rows: int | None = None

    @property
    def latency_us(self) -> float | None:
        return None if self.end_us is None else self.end_us - self.arrival_us

    @property
    def queue_us(self) -> float | None:
        return None if self.start_us is None else self.start_us - self.arrival_us

    @property
    def service_us(self) -> float | None:
        if self.start_us is None or self.end_us is None:
            return None
        return self.end_us - self.start_us

    @property
    def on_time(self) -> bool:
        lat = self.latency_us
        return self.status == "ok" and lat is not None and lat <= self.deadline_us


def bucket_for(rows: int) -> int:
    for b in BUCKETS:
        if rows <= b:
            return b
    raise ValueError(f"{rows} rows exceed the largest bucket {BUCKETS[-1]}")


class BatchExecutor:
    """Runs a batch of rows end to end: host concat -> H2D -> forward -> sync -> D2H.

    Static-shape implementations (torch.compile, CUDA graphs) get one runner per bucket
    and inputs are zero-padded up to the bucket. Padded rows are computed and discarded;
    in the memory-bound small-batch regime that costs little (weights dominate traffic).
    """

    def __init__(self, impl_name: str, weights: ModelWeights, max_rows: int) -> None:
        self.impl = impls.get(impl_name)
        self.weights = weights
        self.device = weights.device
        self.d_model = weights.config.d_model
        self.static = self.impl.static_shape
        if self.static:
            self.runners = {
                b: self.impl.build(weights, rows=b) for b in BUCKETS if b <= bucket_for(max_rows)
            }
        else:
            self.runners = {0: self.impl.build(weights)}

    def run(self, xs: list[torch.Tensor]) -> tuple[list[torch.Tensor], int]:
        x = torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]
        rows = x.shape[0]
        padded = rows
        if self.static:
            padded = bucket_for(rows)
            if padded != rows:
                x = torch.cat([x, x.new_zeros(padded - rows, self.d_model)], dim=0)
            runner = self.runners[padded]
        else:
            runner = self.runners[0]
        y = runner(x.to(self.device, non_blocking=self.device.type == "cuda"))
        y_host = y.to("cpu")  # synchronizes with the device
        synchronize(self.device)
        out = y_host[:rows]
        sizes = [t.shape[0] for t in xs]
        return list(torch.split(out, sizes)), padded


@dataclass
class ReplayResult:
    outcomes: list[Outcome]
    wall_us: float
    busy_us: float
    batches: int
    calibration_us: dict[int, float]
    peak_memory_bytes: int | None
    config: ServerConfig
    impl: str
    errors: list[str] = field(default_factory=list)
    outputs: dict[int, torch.Tensor] | None = None
    #: replay fidelity: lateness of each idle wake-up relative to its target time
    wake_lateness_us: list[float] = field(default_factory=list)


class Server:
    def __init__(
        self,
        impl_name: str,
        weights: ModelWeights,
        config: ServerConfig,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.cfg = config
        self.impl_name = impl_name
        self.exec = BatchExecutor(impl_name, weights, config.max_batch_rows)
        self.clock = clock
        self.calibration = self._calibrate()

    def _calibrate(self) -> dict[int, float]:
        """p90 end-to-end service time per bucket (conservative), used by the dynamic policy."""
        cal: dict[int, float] = {}
        d = self.exec.d_model
        for b in BUCKETS:
            if b > bucket_for(self.cfg.max_batch_rows):
                break
            x = torch.zeros(b, d, dtype=self.exec.weights.dtype)
            for _ in range(5):
                self.exec.run([x])
            samples = []
            for _ in range(self.cfg.calibration_iters):
                t0 = self.clock()
                self.exec.run([x])
                samples.append((self.clock() - t0) / 1e3)
            cal[b] = float(np.percentile(samples, 90))
        return cal

    def predict_service_us(self, rows: int) -> float:
        return self.calibration[bucket_for(rows)]

    # ---- scheduling ---------------------------------------------------------------------

    def _take_batch(self, queue: deque[int], outs: list[Outcome], limit: int) -> list[int]:
        batch: list[int] = []
        rows = 0
        while queue and rows + outs[queue[0]].rows <= limit:
            i = queue.popleft()
            batch.append(i)
            rows += outs[i].rows
        if not batch and queue:  # a single request larger than the limit still gets served
            batch.append(queue.popleft())
        return batch

    def _ready_at(self, queue: deque[int], outs: list[Outcome], now: float) -> float:
        """Earliest time at which the policy wants to start serving the current queue."""
        if not queue:
            return float("inf")
        oldest = outs[queue[0]]
        queued_rows = sum(outs[i].rows for i in queue)
        if self.cfg.policy == "direct":
            return now
        if self.cfg.policy == "static":
            if queued_rows >= self.cfg.batch_rows:
                return now
            return oldest.arrival_us + self.cfg.static_timeout_us
        # dynamic
        if queued_rows >= self.cfg.max_batch_rows:
            return now
        rows = min(queued_rows, self.cfg.max_batch_rows)
        slack_deadline = (
            oldest.arrival_us
            + oldest.deadline_us
            - self.predict_service_us(max(rows, 1))
            - self.cfg.slack_margin_us
        )
        return min(oldest.arrival_us + self.cfg.max_wait_us, slack_deadline)

    def replay(
        self, trace: list[Request], inputs: list[torch.Tensor], keep_outputs: bool = False
    ) -> ReplayResult:
        """Replay ``trace`` open-loop. ``inputs[i]`` is request i's (rows, d_model) host tensor.

        ``keep_outputs`` retains each completed request's output (for correctness tests).
        """
        if len(inputs) != len(trace):
            raise ValueError("one input tensor per request is required")
        outs = [Outcome(r.rid, r.rows, r.arrival_us, r.deadline_us) for r in trace]
        order = sorted(range(len(trace)), key=lambda i: trace[i].arrival_us)
        queue: deque[int] = deque()
        nxt = 0
        busy = 0.0
        batches = 0
        peak = memory_allocated(self.exec.device)
        errors: list[str] = []
        kept: dict[int, torch.Tensor] | None = {} if keep_outputs else None
        lateness: list[float] = []
        t0 = self.clock()

        def now() -> float:
            return (self.clock() - t0) / 1e3

        def admit(t: float) -> None:
            nonlocal nxt
            while nxt < len(order) and trace[order[nxt]].arrival_us <= t:
                i = order[nxt]
                nxt += 1
                if len(queue) >= self.cfg.max_queue:
                    outs[i].status = "rejected"
                else:
                    queue.append(i)

        while nxt < len(order) or queue:
            t = now()
            admit(t)
            if self.cfg.drop_expired:
                while queue and t - outs[queue[0]].arrival_us > outs[queue[0]].deadline_us:
                    outs[queue.popleft()].status = "expired"
            next_arrival = trace[order[nxt]].arrival_us if nxt < len(order) else float("inf")
            ready = self._ready_at(queue, outs, t)
            if not queue or ready > t:
                wake = min(ready, next_arrival)
                if wake == float("inf"):
                    if queue:  # nothing more will arrive: flush (static tail)
                        ready = t
                    else:
                        break
                else:
                    lateness.append(
                        _sleep_until(t0, wake, self.clock, spin_only=self.cfg.idle_wait == "spin")
                    )
                    continue
            limit = self.cfg.batch_rows if self.cfg.policy == "static" else self.cfg.max_batch_rows
            if self.cfg.policy == "direct":
                limit = 1
            batch = self._take_batch(queue, outs, limit)
            start = now()
            try:
                results, padded = self.exec.run([inputs[i] for i in batch])
                status = "ok"
            except Exception as exc:  # isolate failures to this batch
                results, padded, status = [], 0, "error"
                errors.append(f"{type(exc).__name__}: {exc}")
            end = now()
            busy += end - start
            batches += 1
            rows = sum(outs[i].rows for i in batch)
            for i in batch:
                o = outs[i]
                o.status, o.start_us, o.end_us = status, start, end
                o.batch_rows, o.padded_rows = rows, padded
            if kept is not None:
                for i, y in zip(batch, results, strict=False):
                    kept[outs[i].rid] = y
            del results
            mem = memory_allocated(self.exec.device)
            if mem is not None and (peak is None or mem > peak):
                peak = mem
        return ReplayResult(
            outcomes=outs,
            wall_us=now(),
            busy_us=busy,
            batches=batches,
            calibration_us=self.calibration,
            peak_memory_bytes=peak,
            config=self.cfg,
            impl=self.impl_name,
            errors=errors,
            outputs=kept,
            wake_lateness_us=lateness,
        )


#: Wake-up strategy. Measured on macOS: a single long time.sleep (~48 ms) overshot by up to
#: ~3 ms (timer leeway grows with the sleep length), so the replay noticed arrivals late and
#: added artificial queueing delay. Sleeping in chunks of at most CHUNK_US (short sleeps get
#: small leeway) and spinning for the final SPIN_US keeps wake-ups accurate. Every replay
#: reports its wake-up lateness so fidelity is visible, not assumed.
SPIN_US = 1_000.0
CHUNK_US = 1_000.0


def _sleep_until(
    t0: int, target_us: float, clock: Callable[[], int], spin_only: bool = False
) -> float:
    """Wait until ``target_us``; returns how late (µs) the wake-up was."""
    while True:
        now_us = (clock() - t0) / 1e3
        remaining = target_us - now_us
        if remaining <= 0:
            return -remaining
        if not spin_only and remaining > SPIN_US:
            time.sleep(min(remaining - SPIN_US, CHUNK_US) / 1e6)
