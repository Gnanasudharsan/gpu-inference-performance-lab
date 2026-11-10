# Latency-aware serving

Code: [`src/gpulab/serving/`](../src/gpulab/serving). Experiments:
[`scripts/phase4_serving.py`](../scripts/phase4_serving.py) and
[`scripts/idle_gap_interleaved.py`](../scripts/idle_gap_interleaved.py).

## Execution model

- **One device, one worker, non-preemptive service.** Requests are served in FIFO order, in batches.
- **Open-loop arrivals.** Requests arrive on the trace's schedule whether or not the server keeps up.
  A closed-loop load generator, where each client waits for its reply, hides queueing and
  overstates sustainable load, so it is not used.
- **Replay is a single-threaded discrete-event loop.** Before every scheduling decision, all requests
  whose arrival time has passed are admitted with their true arrival timestamps. With non-preemptive
  service on one device this gives exact queueing delays and exact queue occupancy at each arrival.
  It avoids a separate client thread, which would compete for the GIL and add up to 5 ms of
  scheduling jitter (Python's default thread switch interval).
- **A request's service is end to end.** Host-side concatenation, H2D copy, forward pass,
  synchronization, D2H copy and split all count. Payloads are generated before replay and are not
  timed.

## Policies

| Policy | Behavior | Expected regime |
|---|---|---|
| `direct` | Serve the oldest request alone, immediately | Lowest latency at low load; saturates first |
| `static` | Wait until `batch_rows` rows are queued (or a starvation timeout), then serve exactly that batch | High throughput at high load; large added latency at low load |
| `dynamic` | Serve whatever is queued, up to `max_batch_rows`. If not full, wait at most `max_wait_us`, and never beyond *oldest deadline − predicted service time* | Tracks load: batch of 1 when idle, larger batches as the queue builds |

The dynamic policy predicts service time from a per-bucket table measured at server start-up
(`Server.calibration`). Static-shape implementations (`torch.compile` with static shapes, CUDA graphs)
are built once per bucket (1, 2, 4, …, 128 rows), and requests are zero-padded up to the bucket. In the
memory-bound small-batch regime the extra rows cost little, because weight traffic dominates.

## Overload behavior

- **Bounded queue** (`max_queue`). An arrival that finds the queue full is rejected immediately
  (load shedding). Queueing delay is therefore bounded: no unbounded backlog builds up.
- **Expired-request dropping** (`drop_expired`). A request that has already missed its deadline when
  it reaches the head of the queue is dropped instead of consuming device time that on-time requests
  need.
- **Failure isolation.** An exception during a batch marks only that batch's requests as `error`, and
  the server continues. This is tested with an injected fault.

## Metrics

| Metric | Definition |
|---|---|
| `latency_us` | completion − arrival, completed requests only (p50…p99) |
| `queue_us` | service start − arrival |
| `service_us` | service end − service start (the whole batch) |
| `deadline_miss_rate` | 1 − on-time ÷ all requests. **Rejected, expired and failed requests count as misses** |
| `goodput_rows_per_s` | rows from on-time requests ÷ wall time |
| `throughput_rows_per_s` | rows from all completed requests ÷ wall time |
| `utilization` | device busy time ÷ wall time |
| `saturated` | utilization > 0.95 or any rejection |
| `padding_overhead` | padded rows ÷ real rows − 1 (static-shape implementations) |

The **operating envelope** of an (implementation, policy) pair is the highest offered rate at which
the deadline-miss rate stayed within budget (1%). It is reported in `results/phase4/envelope.json`.

## Measured results (Apple M4 GPU via MPS, fp16, 1024×2816×4, 1 row per request)

Source: `results/phase4/` (commit `758dbd6`; idle-gap rerun at `3864043`). Each sweep point is **one
3-second open-loop Poisson replay** with a 10 ms deadline, so these are single runs and point-to-point
noise is visible. The human-readable log is `results/phase4/sweep.log`.

### End-to-end service time is not kernel time

The calibrated end-to-end service time for a 1-row batch was 1.9–9.4 ms (p90, noisy). The replay's
median service time was ~2.0 ms, against ~0.8 ms for the forward pass alone in pipelined mode
(Phase 3). The difference is per-request H2D copy, the synchronizing D2H copy, and the host's
round trip. **This is why a 1.2× kernel speedup barely moves serving metrics here:** the system
bottleneck is the synchronous host↔GPU round trip, not the kernels.

### Saturation and goodput

| impl | `direct`: peak goodput | `dynamic`: peak goodput | `static8`: peak goodput |
|---|---|---|---|
| eager | 251 rows/s (at 400 rps) | 1,158 rows/s (1,600 rps) | 975 rows/s (1,600 rps) |
| metal | 247 rows/s (800 rps) | 1,114 rows/s (1,600 rps) | 1,223 rows/s (1,600 rps) |
| compile | 324 rows/s (600 rps) | 1,503 rows/s (1,600 rps) | 1,058 rows/s (1,600 rps) |

- `direct` saturates at ~400–600 rps (utilization 0.94–0.99). Beyond that, goodput *collapses*
  (to 22–34 rows/s at 1,600 rps), because almost every request waits past its deadline.
- `dynamic` keeps up by batching. Mean batch size grows from 1.1 to 6–8 rows with load, and goodput
  keeps rising through the highest rate tested.
- `static8` misses **every** deadline below 300 rps. Filling 8 rows takes 8 ÷ rate (160 ms at
  50 rps), which is far beyond 10 ms, so requests expire while waiting. This is the textbook failure
  of static batching at low load, and the reason dynamic batching exists.

### Bursty arrivals (300 rps mean, 4× bursts)

| impl | direct miss | static8 miss | dynamic miss |
|---|---|---|---|
| eager | 0.690 | 0.467 | **0.195** |
| metal | 0.570 | 0.385 | **0.135** |
| compile | 0.572 | 0.586 | **0.185** |

Dynamic batching had the lowest miss rate for every implementation under bursts.

### Operating envelope (highest tested rate within the miss budget)

| budget | eager | metal | compile |
|---|---|---|---|
| ≤ 1% misses | direct: 100 rps | none | dynamic: 100 rps |
| ≤ 5% | direct: 100 | none | dynamic: 100, direct: 50 |
| ≤ 10% | direct: 200 | dynamic: 300 | dynamic: 1,600 |

**The envelope is narrow, and the reason was measured, not guessed.** About 7% of synchronous GPU
round trips on this machine include a 3–10+ ms stall (`results/phase3`, latency-mode samples). With a
10 ms deadline, those stalls alone exceed a 1% budget at almost any load. The stall durations are
**not** quantized to the 120 Hz display's 8.3 ms frame period. That hypothesis was tested and *not*
supported, so the root cause is undetermined. Candidates are GPU power-state transitions,
contention with other GPU clients, and command-buffer scheduling.

### Idle-gap sensitivity (interleaved, paired, 30 trials × 20 requests, eager, 8 rows)

| gap before each request | run 1 (`4bed128`) | run 2 (`3864043`, stored) |
|---|---|---|
| 0.5 ms | 1.07× slower [0.93, 1.21] (n.s.) | 1.18× [1.05, 1.34] |
| 2 ms | **1.19× [1.01, 1.39]** | **1.35× [1.18, 1.53]** |
| 10 ms | 1.15× [0.94, 1.42] (n.s.) | 1.19× [1.05, 1.33] |

Idle time before a request makes it slower. The 2 ms effect replicated, and the others reached
significance in one run of two. This settles the question Phase 2 could not, with sequential runs,
and it means sparse real-time traffic pays a latency premium on this GPU.

## Lessons for the operating point

1. **Size the queue from the deadline, not from memory.** With `max_queue = 256` and ~2 ms service, a
   request can wait ~0.5 s. Some p99 values above 100 ms in the sweep are this backlog during stalls.
   A bound of roughly `deadline ÷ service ≈ 5` would shed load earlier and protect p99.
2. **Keep the GPU warm or batch.** Given the measured idle-gap penalty, sparse traffic benefits from
   batching windows or keep-alive work; the tradeoff is power.
3. **Fix the round trip before the kernel.** On this platform, pipelining H2D/compute/D2H across
   requests (or keeping inputs device-resident) would beat any further kernel tuning.

Repeated runs per sweep point and a deadline-derived queue bound are listed as follow-ups in
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).
