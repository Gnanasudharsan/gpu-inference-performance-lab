import time

import numpy as np
import pytest
import torch

from gpulab import impls
from gpulab.correctness import compare
from gpulab.impls.base import Implementation
from gpulab.serving.replay import make_request_inputs, run_serving_experiment, summarize_replay
from gpulab.serving.server import BUCKETS, Server, ServerConfig, bucket_for
from gpulab.serving.trace import TraceSpec, load_trace, make_trace, save_trace
from gpulab.workload import ModelConfig, make_weights

TINY = ModelConfig(d_model=32, d_hidden=64, depth=1)
CPU = torch.device("cpu")


class SleepyImpl(Implementation):
    name = "sleepy_serving"

    def build(self, weights, rows=None):
        inner = impls.get("eager").build(weights)

        def run(x):
            time.sleep(0.002)
            return inner(x)

        return run


class FlakyImpl(Implementation):
    """Fails on batches whose payload sums to > 0 (calibration uses zeros, so it never fails)."""

    name = "flaky"

    def build(self, weights, rows=None):
        inner = impls.get("eager").build(weights)

        def run(x):
            if float(x.sum()) > 0:
                raise RuntimeError("injected fault")
            return inner(x)

        return run


class StaticEager(Implementation):
    """Static-shape wrapper: asserts it only ever sees its bucket size (tests padding)."""

    name = "static_eager"
    static_shape = True

    def build(self, weights, rows=None):
        inner = impls.get("eager").build(weights)

        def run(x):
            assert x.shape[0] == rows, f"built for {rows}, got {x.shape[0]}"
            return inner(x)

        return run


@pytest.fixture(autouse=True)
def _test_impls(monkeypatch):
    for impl in (SleepyImpl(), FlakyImpl(), StaticEager()):
        monkeypatch.setitem(impls._REGISTRY, impl.name, impl)


# ---- traces ---------------------------------------------------------------------------------


def test_trace_is_deterministic_and_seeded():
    spec = TraceSpec(rate_rps=500, duration_s=2, seed=3)
    assert make_trace(spec) == make_trace(spec)
    assert make_trace(spec) != make_trace(TraceSpec(rate_rps=500, duration_s=2, seed=4))


@pytest.mark.parametrize("kind", ["poisson", "constant", "bursty"])
def test_trace_rates(kind):
    spec = TraceSpec(kind=kind, rate_rps=1000, duration_s=4, seed=1)
    tr = make_trace(spec)
    expected = 4000
    tol = (
        4 * np.sqrt(expected)
        if kind == "poisson"
        else (1 if kind == "constant" else 0.2 * expected)
    )
    assert abs(len(tr) - expected) <= tol
    t = [r.arrival_us for r in tr]
    assert t == sorted(t)
    assert t[-1] < 4e6


def test_bursty_is_burstier_than_poisson():
    def cv_of_counts(kind):
        tr = make_trace(TraceSpec(kind=kind, rate_rps=1000, duration_s=10, seed=0))
        counts = np.histogram([r.arrival_us for r in tr], bins=200, range=(0, 10e6))[0]
        return counts.std() / counts.mean()

    assert cv_of_counts("bursty") > 2 * cv_of_counts("poisson")


def test_trace_rows_distribution_and_roundtrip(tmp_path):
    spec = TraceSpec(rate_rps=2000, duration_s=1, rows_choices=(1, 4), rows_probs=(0.75, 0.25))
    tr = make_trace(spec)
    frac4 = np.mean([r.rows == 4 for r in tr])
    assert 0.2 < frac4 < 0.3
    save_trace(tmp_path / "t.jsonl", tr)
    assert load_trace(tmp_path / "t.jsonl") == tr


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "x"},
        {"rate_rps": 0},
        {"rows_choices": (0,)},
        {"rows_probs": (0.5,)},
        {"kind": "bursty", "burst_factor": 10, "burst_fraction": 0.5},
    ],
)
def test_trace_validation(kwargs):
    with pytest.raises(ValueError, match="must"):
        TraceSpec(**kwargs)


# ---- server ---------------------------------------------------------------------------------


def test_server_config_validation_and_buckets():
    with pytest.raises(ValueError, match="policy"):
        ServerConfig(policy="fifo")
    with pytest.raises(ValueError, match="batch_rows"):
        ServerConfig(batch_rows=64, max_batch_rows=32)
    assert bucket_for(1) == 1
    assert bucket_for(3) == 4
    assert bucket_for(33) == 64
    with pytest.raises(ValueError, match="bucket"):
        bucket_for(BUCKETS[-1] + 1)


def _serve(impl, cfg, spec, keep=False):
    w = make_weights(TINY, CPU, torch.float32)
    server = Server(impl, w, ServerConfig(**{**cfg, "calibration_iters": 5}))
    trace = make_trace(spec)
    inputs = make_request_inputs(trace, TINY.d_model, torch.float32)
    res = server.replay(trace, inputs, keep_outputs=keep)
    return server, trace, inputs, res, summarize_replay(res, spec.duration_s)


def test_direct_policy_low_load():
    _, trace, _, res, m = _serve(
        "eager", {"policy": "direct"}, TraceSpec(rate_rps=200, duration_s=0.5, deadline_us=50_000)
    )
    assert m["counts"]["ok"] == len(trace)
    assert m["deadline_miss_rate"] == 0.0
    assert all(o.batch_rows == 1 for o in res.outcomes)
    for o in res.outcomes:
        assert o.start_us >= o.arrival_us - 1e-6
        assert o.latency_us >= o.service_us
    assert 0 < m["utilization"] < 1


def test_static_policy_forms_full_batches():
    _, _, _, _, m = _serve(
        "eager",
        {"policy": "static", "batch_rows": 4, "max_batch_rows": 8},
        TraceSpec(rate_rps=2000, duration_s=0.5, deadline_us=1e6),
    )
    assert m["mean_batch_rows"] > 3.5


def test_dynamic_batches_under_load_not_at_idle():
    spec_hi = TraceSpec(rate_rps=3000, duration_s=0.5, deadline_us=1e6)
    _, _, _, _, hi = _serve("sleepy_serving", {"policy": "dynamic", "max_wait_us": 1000}, spec_hi)
    assert hi["mean_batch_rows"] > 2  # a 2 ms service time at 3k rps forces batching
    spec_lo = TraceSpec(kind="constant", rate_rps=50, duration_s=0.5, deadline_us=1e6)
    _, _, _, _, lo = _serve("eager", {"policy": "dynamic", "max_wait_us": 100}, spec_lo)
    assert lo["mean_batch_rows"] == 1


def test_dynamic_wait_respects_deadline_slack():
    # deadline 5 ms, max_wait 50 ms: the slack bound (deadline - p90 service - margin) must
    # cap waiting at ~4 ms, far below max_wait. The 1 ms margin absorbs rare OS preemption
    # of the replay thread on a shared machine.
    _, _, _, res, _ = _serve(
        "eager",
        {"policy": "dynamic", "max_wait_us": 50_000, "slack_margin_us": 1_000},
        TraceSpec(kind="constant", rate_rps=40, duration_s=0.5, deadline_us=5_000),
    )
    done = [o for o in res.outcomes if o.status == "ok"]
    assert len(done) >= 0.8 * len(res.outcomes), [o.status for o in res.outcomes]
    waits = [o.queue_us for o in done]
    assert max(waits) < 6_000
    assert float(np.median(waits)) > 2_000  # it did wait for batching opportunities
    assert float(np.median(res.wake_lateness_us)) < 100  # replay fidelity


def test_bounded_queue_sheds_load_and_counts_misses():
    _, trace, _, res, m = _serve(
        "sleepy_serving",
        {"policy": "direct", "max_queue": 4, "drop_expired": False},
        TraceSpec(rate_rps=4000, duration_s=0.3, deadline_us=1e6),
    )
    assert m["counts"]["rejected"] > 0
    assert m["saturated"]
    done = [o for o in res.outcomes if o.status == "ok"]
    # bounded queue => bounded waiting: at most max_queue services ahead, plus the one in
    # progress at arrival (measured service times, not an assumed sleep duration)
    worst_service = max(o.service_us for o in done)
    assert max(o.queue_us for o in done) <= (4 + 1) * worst_service + 1_000
    # rejected requests are misses; equality is exact in math but not in floating point
    assert m["deadline_miss_rate"] >= m["counts"]["rejected"] / len(trace) - 1e-12


def test_expired_requests_are_dropped():
    _, _, _, _, m = _serve(
        "sleepy_serving",
        {"policy": "direct", "drop_expired": True},
        TraceSpec(rate_rps=3000, duration_s=0.3, deadline_us=3000),
    )
    assert m["counts"]["expired"] > 0
    assert m["deadline_miss_rate"] > 0


def test_failures_are_isolated_to_their_batch():
    _, trace, _, res, m = _serve(
        "flaky", {"policy": "direct"}, TraceSpec(rate_rps=200, duration_s=0.5, deadline_us=1e6)
    )
    assert m["counts"]["error"] > 0
    assert m["counts"]["ok"] > 0
    assert m["counts"]["ok"] + m["counts"]["error"] == len(trace)
    assert "injected fault" in res.errors[0]


@pytest.mark.parametrize("impl", ["eager", "static_eager"])
def test_batching_and_padding_preserve_outputs(impl):
    server, trace, inputs, res, _ = _serve(
        impl,
        {"policy": "dynamic", "max_wait_us": 2000},
        TraceSpec(rate_rps=3000, duration_s=0.3, deadline_us=1e6, rows_choices=(1, 3)),
        keep=True,
    )
    assert max(o.batch_rows for o in res.outcomes) > 1
    single = impls.get("eager").build(server.exec.weights)
    for r, x in zip(trace, inputs, strict=True):
        y = res.outputs[r.rid]
        assert y.shape == x.shape
        assert compare(y, single(x).double(), torch.float32).passed
    if impl == "static_eager":
        assert all(o.padded_rows in BUCKETS for o in res.outcomes)


def test_run_serving_experiment_record():
    rec, _ = run_serving_experiment(
        "eager",
        TINY,
        CPU,
        torch.float32,
        ServerConfig(policy="dynamic", calibration_iters=3),
        TraceSpec(rate_rps=300, duration_s=0.3),
    )
    assert rec.kind == "serving"
    assert rec.metrics["requests"] > 0
    assert rec.params["server"]["policy"] == "dynamic"
    assert rec.metrics["latency_us"]["p99"] >= rec.metrics["latency_us"]["p50"]
