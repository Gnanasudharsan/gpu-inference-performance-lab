import json

from gpulab.cli import main
from gpulab.provenance import collect_provenance, format_report


def test_provenance_has_cpu_accelerator_and_versions():
    p = collect_provenance()
    backends = [a.backend for a in p.accelerators]
    assert "cpu" in backends
    assert p.torch
    assert p.python
    assert p.numpy
    # hostnames are never stored in clear text
    assert len(p.hostname_hash) == 12


def test_provenance_is_json_serialisable():
    d = collect_provenance().to_dict()
    assert json.loads(json.dumps(d))["torch"] == d["torch"]


def test_format_report_mentions_every_accelerator():
    p = collect_provenance()
    text = format_report(p)
    for a in p.accelerators:
        assert a.name in text


def test_cli_env_json(capsys):
    assert main(["env", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "accelerators" in out
