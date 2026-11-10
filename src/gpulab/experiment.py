"""Config-driven experiment runner.

An experiment is a TOML file with ``[[bench]]`` groups (interleaved, paired comparisons)
and ``[[serving]]`` groups (open-loop replay sweeps). The runner executes every group that
is runnable on this machine, records *why* any group or implementation was skipped, and
writes::

    <out>/bench.jsonl      benchmark records (one per impl x rows x mode)
    <out>/serving.jsonl    serving records (one per impl x policy x rate)
    <out>/manifest.json    config, timing, provenance, skipped groups/impls

Nothing is estimated or filled in for skipped work: it is simply absent and listed.
"""

from __future__ import annotations

import json
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from gpulab import impls
from gpulab.bench import BenchConfig, run_interleaved
from gpulab.device import available_devices, parse_dtype
from gpulab.provenance import collect_provenance
from gpulab.records import append_jsonl
from gpulab.serving.replay import run_serving_experiment
from gpulab.serving.server import Server, ServerConfig
from gpulab.serving.trace import TraceSpec
from gpulab.workload import ModelConfig, make_weights

POLICY_PRESETS: dict[str, dict[str, Any]] = {
    "direct": {"policy": "direct"},
    "static8": {"policy": "static", "batch_rows": 8, "max_batch_rows": 32},
    "dynamic": {"policy": "dynamic", "max_wait_us": 500.0, "max_batch_rows": 32},
}


@dataclass
class BenchGroup:
    device: str
    dtype: str
    impls: list[str]
    rows: list[int]
    modes: list[str] = field(default_factory=lambda: ["latency", "throughput"])
    d_model: int = 1024
    d_hidden: int = 2816
    depth: int = 4
    iters: int = 400
    trials: int = 20
    burst: int = 50
    warmup_s: float = 1.0
    deadline_us: float | None = None


@dataclass
class ServingGroup:
    device: str
    dtype: str
    impls: list[str]
    policies: list[str]
    rates: list[float]
    arrival: str = "poisson"
    duration_s: float = 3.0
    deadline_us: float = 10_000.0
    d_model: int = 1024
    d_hidden: int = 2816
    depth: int = 4


@dataclass
class Experiment:
    name: str
    description: str = ""
    bench: list[BenchGroup] = field(default_factory=list)
    serving: list[ServingGroup] = field(default_factory=list)


def load_experiment(path: str | Path) -> Experiment:
    raw = tomllib.loads(Path(path).read_text())
    unknown = set(raw) - {"name", "description", "bench", "serving"}
    if unknown:
        raise ValueError(f"unknown top-level keys: {sorted(unknown)}")
    exp = Experiment(
        name=raw["name"],
        description=raw.get("description", ""),
        bench=[BenchGroup(**g) for g in raw.get("bench", [])],
        serving=[ServingGroup(**g) for g in raw.get("serving", [])],
    )
    for g in exp.serving:
        bad = [p for p in g.policies if p not in POLICY_PRESETS]
        if bad:
            raise ValueError(f"unknown policies {bad}; have {sorted(POLICY_PRESETS)}")
    for bg in exp.bench:
        if bg.impls[0] != "eager" and "eager" not in bg.impls:
            raise ValueError("bench groups must include the eager baseline for paired speedups")
    return exp


def _runnable(
    names: list[str], device: str, dtype: torch.dtype
) -> tuple[list[str], dict[str, str]]:
    ok, skipped = [], {}
    for n in names:
        reason = impls.get(n).unsupported_reason(torch.device(device), dtype)
        if reason:
            skipped[n] = reason
        else:
            ok.append(n)
    return ok, skipped


def run_experiment(exp: Experiment, out_dir: str | Path, log: Any = print) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("bench.jsonl", "serving.jsonl"):
        (out / f).unlink(missing_ok=True)
    manifest: dict[str, Any] = {
        "name": exp.name,
        "description": exp.description,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": collect_provenance().to_dict(),
        "skipped": [],
        "counts": {"bench": 0, "serving": 0},
    }
    have = available_devices()
    t0 = time.time()
    for i, g in enumerate(exp.bench):
        if g.device not in have:
            manifest["skipped"].append({"group": f"bench[{i}]", "reason": f"no {g.device} device"})
            log(f"skip bench[{i}]: no {g.device} device")
            continue
        dt = parse_dtype(g.dtype)
        names, skipped = _runnable(g.impls, g.device, dt)
        for n, why in skipped.items():
            manifest["skipped"].append({"group": f"bench[{i}]", "impl": n, "reason": why})
        model = ModelConfig(d_model=g.d_model, d_hidden=g.d_hidden, depth=g.depth)
        cfg = BenchConfig(
            iters=g.iters,
            trials=g.trials,
            burst=g.burst,
            warmup_s=g.warmup_s,
            deadline_us=g.deadline_us,
        )
        for rows in g.rows:
            for mode in g.modes:
                recs = run_interleaved(
                    names, model, rows, torch.device(g.device), dt, cfg, mode=mode
                )
                append_jsonl(out / "bench.jsonl", recs)
                manifest["counts"]["bench"] += len(recs)
                log(
                    f"bench[{i}] {g.device}/{g.dtype} {model.d_model} rows={rows} {mode}: "
                    + ", ".join(f"{r.impl}={r.metrics['host_us']['p50']:.0f}us" for r in recs)
                )
    for i, sg in enumerate(exp.serving):
        if sg.device not in have:
            manifest["skipped"].append(
                {"group": f"serving[{i}]", "reason": f"no {sg.device} device"}
            )
            continue
        dt = parse_dtype(sg.dtype)
        names, skipped = _runnable(sg.impls, sg.device, dt)
        for n, why in skipped.items():
            manifest["skipped"].append({"group": f"serving[{i}]", "impl": n, "reason": why})
        model = ModelConfig(d_model=sg.d_model, d_hidden=sg.d_hidden, depth=sg.depth)
        dev = torch.device(sg.device)
        for name in names:
            server = Server(
                name, make_weights(model, dev, dt), ServerConfig(**POLICY_PRESETS["dynamic"])
            )
            for pol in sg.policies:
                pcfg = ServerConfig(**POLICY_PRESETS[pol])
                for rate in sg.rates:
                    spec = TraceSpec(
                        kind=sg.arrival,
                        rate_rps=rate,
                        duration_s=sg.duration_s,
                        seed=int(rate),
                        deadline_us=sg.deadline_us,
                    )
                    rec, _ = run_serving_experiment(name, model, dev, dt, pcfg, spec, server)
                    rec.params["policy_name"] = pol
                    append_jsonl(out / "serving.jsonl", [rec])
                    manifest["counts"]["serving"] += 1
                    m = rec.metrics
                    log(
                        f"serving[{i}] {name}/{pol} {rate:.0f} rps: "
                        f"miss {m['deadline_miss_rate']:.3f} "
                        f"goodput {m['goodput_rows_per_s']:.0f} rows/s"
                    )
            del server
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
