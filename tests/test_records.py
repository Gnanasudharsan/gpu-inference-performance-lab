import json

import pytest

from gpulab.cli import main
from gpulab.records import ExperimentRecord, append_jsonl, flatten, read_jsonl, write_parquet


def _rec(**kw):
    base = dict(
        kind="benchmark",
        impl="eager",
        device="cpu",
        device_name="cpu",
        dtype="float32",
        rows=1,
        model={"d_model": 8},
        metrics={"p50_us": 1.0, "samples": [1, 2]},
    )
    base.update(kw)
    return ExperimentRecord(**base)


def test_record_roundtrip_jsonl(tmp_path):
    p = tmp_path / "r.jsonl"
    recs = [_rec(), _rec(rows=2)]
    assert append_jsonl(p, recs) == 2
    back = read_jsonl(p)
    assert [r.to_dict() for r in back] == [r.to_dict() for r in recs]
    assert back[0].record_id.startswith("ben-")
    assert back[0].record_id != back[1].record_id
    assert back[0].provenance["torch"]


def test_record_rejects_unknown_evidence():
    with pytest.raises(ValueError, match="evidence"):
        _rec(evidence="guessed")


def test_flatten_and_parquet(tmp_path):
    row = flatten(_rec())
    assert row["metrics.p50_us"] == 1.0
    assert json.loads(row["metrics.samples"]) == [1, 2]
    assert row["model.d_model"] == 8
    pd = pytest.importorskip("pandas")
    path = write_parquet(tmp_path / "r.parquet", [_rec(), _rec(rows=4)])
    df = pd.read_parquet(path)
    assert list(df["rows"]) == [1, 4]


def test_cli_run_saves_passing_record(tmp_path, capsys):
    out = tmp_path / "c.jsonl"
    rc = main(
        [
            "run",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--rows",
            "2",
            "--d-model",
            "32",
            "--d-hidden",
            "64",
            "--depth",
            "1",
            "--save",
            str(out),
        ]
    )
    assert rc == 0
    assert "PASS" in capsys.readouterr().out
    (rec,) = read_jsonl(out)
    assert rec.kind == "correctness"
    assert rec.correctness["passed"] is True


def test_cli_run_unknown_impl():
    with pytest.raises(KeyError):
        main(["run", "--impl", "nope", "--device", "cpu"])
