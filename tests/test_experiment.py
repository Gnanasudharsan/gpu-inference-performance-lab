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
    import torch

    has_cuda = torch.cuda.is_available()
    # CPU group: 2 impls x 2 modes; the CUDA group (1 impl x 2 modes) runs only where CUDA exists
    assert manifest["counts"] == {"bench": 4 + (2 if has_cuda else 0), "serving": 1}
    reasons = {(s.get("impl"), s["reason"]) for s in manifest["skipped"]}
    assert ("metal", "Metal kernels run only on Apple GPUs (MPS)") in reasons
    assert has_cuda or any("no cuda device" in r for _, r in reasons)
    # analysis over a results tree laid out like the repository's
    results = tmp_path / "results"
    (results / "phase3").mkdir(parents=True)
    (results / "phase4").mkdir(parents=True)
    (out / "bench.jsonl").rename(results / "phase3" / "bench_t.jsonl")
    (out / "serving.jsonl").rename(results / "phase4" / "serving_sweep.jsonl")
    s = build_summary(results)
    assert len(s["benchmarks"]) == 4 + (2 if has_cuda else 0)
    assert {c["mode"] for c in s["comparisons"]} == {"latency", "throughput"}
    assert s["serving"][0]["policy"] == "direct"
    assert has_cuda != any("NVIDIA" in u for u in s["unavailable"])
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


def test_summary_ingests_gpu_sessions(tmp_path):
    """results/gpu/<run>/gpulab_results/* is merged; the NVIDIA-unavailable note depends on it."""
    from gpulab.records import append_jsonl, read_jsonl

    cfg = tmp_path / "t.toml"
    cfg.write_text(CONFIG)
    run_experiment(load_experiment(cfg), tmp_path / "out", log=lambda *_: None)
    results = tmp_path / "results"
    (results / "phase3").mkdir(parents=True)
    run_dir = results / "gpu" / "kaggle-t4-test" / "gpulab_results"
    (run_dir / "experiment").mkdir(parents=True)
    # TEST FIXTURE: a CPU record relabeled as cuda, only to exercise the ingestion logic.
    recs = read_jsonl(tmp_path / "out" / "bench.jsonl")
    for r in recs:
        r.device = "cuda"
    append_jsonl(run_dir / "experiment" / "bench.jsonl", recs)
    (run_dir / "session.json").write_text(
        json.dumps(
            {
                "steps": [
                    {"step": "correctness gate", "status": "ok", "seconds": 1.0},
                    {"step": "nsight systems", "status": "skipped", "reason": "nsys not installed"},
                ]
            }
        )
    )
    (run_dir / "env.json").write_text(
        json.dumps({"git_commit": "abc", "accelerators": [{"backend": "cuda", "name": "Tesla T4"}]})
    )
    s = build_summary(results)
    assert s["devices"] == ["cuda"]
    assert s["gpu_sessions"][0]["gpu"] == "Tesla T4"
    assert s["gpu_sessions"][0]["steps"][1]["status"] == "skipped"
    assert not any("NVIDIA GPU results" in u for u in s["unavailable"])
    assert any("Nsight" in u for u in s["unavailable"])  # nsys was skipped


def test_summary_reports_only_the_latest_gate_passing_gpu_session(tmp_path):
    """A failed gate's numbers are never reported, and two sessions are never pooled."""
    from gpulab.records import append_jsonl, read_jsonl

    cfg = tmp_path / "t.toml"
    cfg.write_text(CONFIG)
    run_experiment(load_experiment(cfg), tmp_path / "out", log=lambda *_: None)
    recs = read_jsonl(tmp_path / "out" / "bench.jsonl")
    results = tmp_path / "results"
    (results / "phase3").mkdir(parents=True)

    def session(name: str, gate: str, n_recs: int, profile: dict | None = None) -> None:
        d = results / "gpu" / name / "gpulab_results"
        (d / "experiment").mkdir(parents=True)
        # TEST FIXTURE: CPU records relabeled as cuda, only to exercise the ingestion logic.
        for r in recs[:n_recs]:
            r.device = "cuda"
        append_jsonl(d / "experiment" / "bench.jsonl", recs[:n_recs])
        (d / "session.json").write_text(
            json.dumps({"steps": [{"step": "correctness gate", "status": gate}]})
        )
        if profile is not None:
            (d / "profile_eager_r8_cuda.json").write_text(json.dumps(profile))

    old_profile = {
        "impl": "eager",
        "rows": 8,
        "dispatches_per_forward": 60,
        "cuda_kernels_per_forward": 61,
        "cuda_kernel_us_per_forward": 2828.0,
    }
    session("t4-1-failed", "failed", 4)
    session("t4-2-passed", "ok", 2, profile=old_profile)
    session("t4-3-failed", "failed", 4)
    s = build_summary(results)
    assert len(s["benchmarks"]) == 2  # only t4-2's records
    flags = {g["run"]: g["measurements_reported"] for g in s["gpu_sessions"]}
    assert flags == {"t4-1-failed": False, "t4-2-passed": True, "t4-3-failed": False}
    # pre-fix profile: kernel count/time counted the record_function span -> withheld
    prof = s["kernels"]["cuda_profiles"]["eager/r8"]
    assert prof["aten"] == 60
    assert prof["cuda_kernels"] is None
    assert prof["cuda_kernel_us"] is None

    session(
        "t4-4-passed",
        "ok",
        3,
        profile={**old_profile, "annotations_excluded": True, "cuda_kernels_per_forward": 60},
    )
    s = build_summary(results)
    assert len(s["benchmarks"]) == 3  # the newest passing session replaces t4-2
    assert s["kernels"]["cuda_profiles"]["eager/r8"]["cuda_kernels"] == 60
