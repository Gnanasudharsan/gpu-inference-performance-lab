import numpy as np
import pytest

from gpulab.report import paired_speedup
from gpulab.stats import (
    bootstrap_ci,
    deadline_goodput,
    lag1_autocorrelation,
    percentile,
    repeatability,
    speedup,
    summarize,
)


def test_percentiles_match_numpy():
    a = np.arange(1, 101, dtype=float)
    s = summarize(a)
    assert s.n == 100
    assert s.min == 1
    assert s.max == 100
    for q in (50, 90, 95, 99):
        assert getattr(s, f"p{q}") == pytest.approx(np.percentile(a, q))
    assert percentile([5.0], 99) == 5.0
    with pytest.raises(ValueError, match="no samples"):
        summarize([])


def test_bootstrap_ci_covers_true_median():
    rng = np.random.default_rng(1)
    a = rng.normal(100.0, 5.0, size=4000)
    lo, hi = bootstrap_ci(a, 50)
    assert lo < 100.0 < hi
    assert hi - lo < 1.0


def test_ci_only_when_sample_size_justifies_it():
    assert summarize(np.ones(10)).p50_ci is None
    s = summarize(np.random.default_rng(0).normal(size=200))
    assert s.p50_ci is not None
    assert s.p99_ci is None  # p99 needs >= 1000 samples
    assert summarize(np.random.default_rng(0).normal(size=1000)).p99_ci is not None


def test_autocorrelation_flags_clustered_samples():
    rng = np.random.default_rng(0)
    iid = rng.normal(size=5000)
    assert abs(lag1_autocorrelation(iid)) < 0.05
    ar = np.zeros(5000)
    for i in range(1, ar.size):
        ar[i] = 0.9 * ar[i - 1] + rng.normal()
    assert lag1_autocorrelation(ar) > 0.8
    assert summarize(iid).ci_reliable
    assert not summarize(ar + 100).ci_reliable
    assert lag1_autocorrelation(np.ones(10)) == 0.0


def test_deadline_goodput_exact():
    lat = [100.0, 200.0, 300.0, 400.0]  # 1 ms busy in total
    g = deadline_goodput(lat, rows_per_request=2, deadline_us=250.0)
    assert g["on_time_fraction"] == 0.5
    assert g["miss_rate"] == 0.5
    assert g["throughput_rows_per_s"] == pytest.approx(8 / 1e-3)
    assert g["goodput_rows_per_s"] == pytest.approx(4 / 1e-3)
    with pytest.raises(ValueError, match="deadline"):
        deadline_goodput(lat, 1, 0)


def test_repeatability_and_speedup():
    r = repeatability([100.0, 100.0, 100.0])
    assert r["cv"] == 0.0
    assert r["max_rel_spread"] == 0.0
    r = repeatability([90.0, 110.0])
    assert r["cv"] == pytest.approx(np.std([90, 110], ddof=1) / 100)
    assert speedup(200.0, 100.0) == 2.0
    with pytest.raises(ValueError, match="positive"):
        speedup(0.0, 1.0)


def test_paired_speedup_geomean():
    s = paired_speedup([200.0, 400.0], [100.0, 100.0])
    assert s["geomean"] == pytest.approx(np.sqrt(2 * 4))
    assert s["min"] == 2.0
    assert s["max"] == 4.0
    with pytest.raises(ValueError, match="equal"):
        paired_speedup([1.0], [1.0, 2.0])
