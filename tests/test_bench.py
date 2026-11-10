import time

import pytest
import torch

from gpulab import impls
from gpulab.bench import BenchConfig, run_benchmark, run_interleaved
from gpulab.cli import main
from gpulab.impls.base import Implementation
from gpulab.profiling import profile_forward
from gpulab.records import read_jsonl
from gpulab.report import benchmark_table, paired_speedup
from gpulab.workload import ModelConfig

TINY = ModelConfig(d_model=32, d_hidden=64, depth=2)
FAST = BenchConfig(
    iters=30,
    trials=3,
    warmup_iters=2,
    warmup_s=0.0,
    device_iters=10,
    burst=5,
    bursts=3,
    deadline_us=1e6,
)


class SleepImpl(Implementation):
    """Correct (delegates to eager) but adds a fixed 2 ms host delay."""

    name = "sleepy"

    def build(self, weights, rows=None):
        inner = impls.get("eager").build(weights)

        def run(x):
            time.sleep(0.002)
            return inner(x)

        return run


class BrokenImpl(Implementation):
    name = "broken"

    def build(self, weights, rows=None):
        return lambda x: x * 0


@pytest.fixture
def extra_impls(monkeypatch):
    monkeypatch.setitem(impls._REGISTRY, "sleepy", SleepImpl())
    monkeypatch.setitem(impls._REGISTRY, "broken", BrokenImpl())


def test_config_validation():
    with pytest.raises(ValueError, match="iters"):
        BenchConfig(iters=2, trials=3)
    with pytest.raises(ValueError, match="burst"):
        BenchConfig(burst=0)
    with pytest.raises(ValueError, match="idle_gap"):
        BenchConfig(idle_gap_us=-1)


def test_benchmark_all_modes(device):
    dtype = torch.float32
    recs = run_benchmark(
        "eager", TINY, 4, torch.device(device), dtype, FAST, modes=("latency", "e2e", "throughput")
    )
    assert [r.metrics["mode"] for r in recs] == ["latency", "e2e", "throughput"]
    lat, e2e, thr = recs
    assert lat.metrics["host_us"]["n"] == 30
    assert len(lat.metrics["samples_host_us"]) == 30
    assert len(lat.metrics["trial_medians_us"]) == 3
    assert thr.metrics["host_us"]["n"] == 3  # one sample per burst
    assert lat.correctness["passed"]
    assert lat.metrics["goodput"]["on_time_fraction"] == 1.0
    assert lat.params["workload"]["rows"] == 4
    if device == "cpu":
        assert lat.metrics["device_us"] is None
        assert e2e.metrics["stages_us"] is None
    else:
        assert lat.metrics["device_us"]["n"] == 10
        assert set(e2e.metrics["stages_us"]) == {"h2d", "compute", "d2h"}
    for r in recs:
        assert r.metrics["host_us"]["min"] > 0


def test_host_timing_measures_real_delay(extra_impls):
    (rec,) = run_benchmark("sleepy", TINY, 2, torch.device("cpu"), torch.float32, FAST)
    p50 = rec.metrics["host_us"]["p50"]
    assert 2000 <= p50 < 20000


def test_correctness_gate_blocks_broken_impl(extra_impls):
    with pytest.raises(AssertionError, match="failed correctness"):
        run_benchmark("broken", TINY, 2, torch.device("cpu"), torch.float32, FAST)


def test_interleaved_pairs_trials(extra_impls):
    recs = run_interleaved(["eager", "sleepy"], TINY, 2, torch.device("cpu"), torch.float32, FAST)
    base, slow = recs
    assert base.params["interleaved"]["group"] == slow.params["interleaved"]["group"]
    assert len(base.metrics["trial_medians_us"]) == len(slow.metrics["trial_medians_us"]) == 3
    s = paired_speedup(base.metrics["trial_medians_us"], slow.metrics["trial_medians_us"])
    assert s["max"] < 1.0  # the sleepy impl is slower in every trial
    thr = run_interleaved(
        ["eager", "sleepy"], TINY, 2, torch.device("cpu"), torch.float32, FAST, mode="throughput"
    )
    assert len(thr[0].metrics["trial_medians_us"]) == FAST.trials
    assert thr[1].metrics["host_us"]["min"] > 2000  # 2 ms sleep per request
    with pytest.raises(ValueError, match="interleaved mode"):
        run_interleaved(["eager"], TINY, 2, torch.device("cpu"), torch.float32, FAST, mode="e2e")
    table = benchmark_table(recs)
    assert "sleepy" in table
    assert "×" in table
    with pytest.raises(ValueError, match="unique"):
        run_interleaved(["eager", "eager"], TINY, 2, torch.device("cpu"), torch.float32, FAST)


def test_profile_dispatches_scale_with_depth():
    a = profile_forward(
        "eager", ModelConfig(32, 64, depth=2), 2, torch.device("cpu"), torch.float32, iters=3
    )
    b = profile_forward(
        "eager", ModelConfig(32, 64, depth=4), 2, torch.device("cpu"), torch.float32, iters=3
    )
    assert a["dispatches_per_forward"] > 0
    assert b["dispatches_per_forward"] == pytest.approx(2 * a["dispatches_per_forward"])


def test_cli_bench_and_report(tmp_path, capsys):
    out = tmp_path / "b.jsonl"
    rc = main(
        [
            "bench",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--rows",
            "1",
            "2",
            "--d-model",
            "32",
            "--d-hidden",
            "64",
            "--depth",
            "1",
            "--iters",
            "9",
            "--trials",
            "3",
            "--warmup-s",
            "0",
            "--save",
            str(out),
        ]
    )
    assert rc == 0
    assert len(read_jsonl(out)) == 2
    md = tmp_path / "r.md"
    pq = tmp_path / "r.parquet"
    assert main(["report", str(out), "--out", str(md), "--parquet", str(pq)]) == 0
    assert "| cpu | float32 | eager | 1 |" in md.read_text()
    assert pq.exists()
