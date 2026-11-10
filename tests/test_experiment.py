import json

import pytest

from gpulab.analysis import paired_ratio_ci, verdict
from gpulab.analysis_report import render
from gpulab.cli import main
from gpulab.experiment import load_experiment, run_experiment
from gpulab.summary import build_summary

CONFIG = """
name = "t"
[[bench]]
device = "cpu"
dtype = "float32"
impls = ["eager", "fused_weights", "metal"]
rows = [2]
modes = ["latency", "throughput"]
d_model = 32
d_hidden = 64
depth = 1
iters = 6
trials = 3
burst = 3
warmup_s = 0.0

[[bench]]
device = "cuda"
dtype = "float16"
impls = ["eager"]
rows = [1]

[[serving]]
device = "cpu"
dtype = "float32"
impls = ["eager"]
policies = ["direct"]
rates = [200]
duration_s = 0.2
d_model = 32
d_hidden = 64
depth = 1
"""


def test_paired_ratio_ci_and_verdict():
    g, lo, hi = paired_ratio_ci([2.0, 2.1, 1.9, 2.0], [1.0, 1.0, 1.0, 1.0])
    assert lo <= g <= hi
    assert verdict(lo, hi) == "faster"
    assert verdict(0.5, 0.9) == "slower"
    assert verdict(0.9, 1.1) == "no significant difference"
    with pytest.raises(ValueError, match="paired"):
        paired_ratio_ci([1.0], [1.0])


def test_load_experiment_validation(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('name = "x"\nfoo = 1\n')
    with pytest.raises(ValueError, match="unknown top-level"):
        load_experiment(p)
    p.write_text(
        'name="x"\n[[serving]]\ndevice="cpu"\ndtype="float32"\nimpls=["eager"]\n'
        'policies=["fifo"]\nrates=[1]\n'
    )
    with pytest.raises(ValueError, match="unknown policies"):
        load_experiment(p)


def test_run_experiment_records_skips_and_analyzes(tmp_path):
    cfg = tmp_path / "t.toml"
    cfg.write_text(CONFIG)
    out = tmp_path / "out"
    manifest = run_experiment(load_experiment(cfg), out, log=lambda *_: None)
    assert manifest["counts"] == {"bench": 4, "serving": 1}  # 2 impls x 2 modes; 1 serving
    reasons = {(s.get("impl"), s["reason"]) for s in manifest["skipped"]}
    assert ("metal", "Metal kernels run only on Apple GPUs (MPS)") in reasons
    assert any("no cuda device" in r for _, r in reasons) or __import__("torch").cuda.is_available()
    # analysis over a results tree laid out like the repository's
    results = tmp_path / "results"
    (results / "phase3").mkdir(parents=True)
    (results / "phase4").mkdir(parents=True)
    (out / "bench.jsonl").rename(results / "phase3" / "bench_t.jsonl")
    (out / "serving.jsonl").rename(results / "phase4" / "serving_sweep.jsonl")
    s = build_summary(results)
    assert len(s["benchmarks"]) == 4
    assert {c["mode"] for c in s["comparisons"]} == {"latency", "throughput"}
    assert s["serving"][0]["policy"] == "direct"
    assert any("NVIDIA" in u for u in s["unavailable"])
    md = render(s)
    assert "## Regressions" in md
    assert "fused_weights" in md
    json.dumps(s)  # serialisable


def test_cli_analyze(tmp_path, capsys):
    results = tmp_path / "results"
    (results / "phase3").mkdir(parents=True)
    rc = main(
        [
            "analyze",
            "--results",
            str(results),
            "--summary",
            str(tmp_path / "s.json"),
            "--report",
            str(tmp_path / "a.md"),
        ]
    )
    assert rc == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert "phase3 benchmarks" in s["unavailable"]
