"""Compile measured artifacts into one JSON dataset (consumed by the dashboard and reports).

Everything in the summary is either copied from records (``measured``), computed from them
(``derived``), or taken from specifications (``theoretical``); each section says which.
Missing inputs are listed under ``unavailable`` rather than filled in.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from gpulab.analysis import comparisons, roofline
from gpulab.records import ExperimentRecord, read_jsonl

QUANTILES = np.linspace(0, 100, 101)


def _read_all(paths: list[Path]) -> list[ExperimentRecord]:
    return [r for p in paths if p.exists() for r in read_jsonl(p)]


def _bench_entry(r: ExperimentRecord) -> dict[str, Any]:
    m = r.metrics
    host = m.get("host_us") or {}
    samples = m.get("samples_host_us") or []
    entry: dict[str, Any] = {
        "id": r.record_id,
        "impl": r.impl,
        "device": r.device,
        "device_name": r.device_name,
        "dtype": r.dtype,
        "rows": r.rows,
        "d_model": r.model["d_model"],
        "d_hidden": r.model["d_hidden"],
        "depth": r.model["depth"],
        "mode": m["mode"],
        "group": (r.params.get("interleaved") or {}).get("group"),
        "host_us": {
            k: host.get(k) for k in ("n", "p50", "p90", "p95", "p99", "min", "max", "mean")
        },
        "p50_ci": host.get("p50_ci"),
        "ci_reliable": host.get("ci_reliable"),
        "lag1_autocorr": host.get("lag1_autocorr"),
        "device_p50_us": (m.get("device_us") or {}).get("p50"),
        "throughput_rows_per_s": m.get("throughput_rows_per_s"),
        "trial_cv": (m.get("repeatability") or {}).get("cv"),
        "trials": len(m.get("trial_medians_us") or []),
        "build_s": m.get("build_s"),
        "first_call_us": m.get("first_call_us"),
        "memory": m.get("memory"),
        "correctness": (r.correctness or {}).get("rel_l2"),
        "correct": (r.correctness or {}).get("passed"),
        "roofline": roofline(r) if m["mode"] == "throughput" else None,
        "git_commit": r.provenance.get("git_commit"),
        "git_dirty": r.provenance.get("git_dirty"),
        "created_utc": r.created_utc,
    }
    if len(samples) >= 20:
        entry["quantiles_host_us"] = [round(float(v), 2) for v in np.percentile(samples, QUANTILES)]
    return entry


def _serving_entry(r: ExperimentRecord) -> dict[str, Any]:
    m = r.metrics
    lat = m.get("latency_us") or {}
    q = m.get("queue_us") or {}
    svc = m.get("service_us") or {}
    return {
        "id": r.record_id,
        "impl": r.impl,
        "device": r.device,
        "dtype": r.dtype,
        "policy": r.params.get("policy_name") or r.params["server"]["policy"],
        "arrival": r.params["trace"]["kind"],
        "rate_rps": r.params["trace"]["rate_rps"],
        "deadline_us": r.params["trace"]["deadline_us"],
        "requests": m["requests"],
        "counts": m["counts"],
        "latency_us": {k: lat.get(k) for k in ("p50", "p95", "p99", "max")},
        "queue_p50_us": q.get("p50"),
        "queue_p99_us": q.get("p99"),
        "service_p50_us": svc.get("p50"),
        "deadline_miss_rate": m["deadline_miss_rate"],
        "goodput_rows_per_s": m["goodput_rows_per_s"],
        "throughput_rows_per_s": m["throughput_rows_per_s"],
        "utilization": m["utilization"],
        "mean_batch_rows": m["mean_batch_rows"],
        "padding_overhead": m.get("padding_overhead"),
        "saturated": m["saturated"],
        "wake_lateness_p99_us": (m.get("wake_lateness_us") or {}).get("p99"),
        "git_commit": r.provenance.get("git_commit"),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def build_summary(results: Path) -> dict[str, Any]:
    p3, p4, p1, p2 = (results / d for d in ("phase3", "phase4", "phase1", "phase2"))
    bench = _read_all(sorted(p3.glob("bench_*.jsonl")))
    serving = _read_all([p4 / "serving_sweep.jsonl", p4 / "serving_bursty.jsonl"])
    correctness = _read_all([p1 / "correctness.jsonl"])
    unavailable = []
    if not bench:
        unavailable.append("phase3 benchmarks")
    if not serving:
        unavailable.append("phase4 serving")
    profiles = {}
    for f in sorted(p3.glob("profile_*_mps.json")):
        d = json.loads(f.read_text())
        profiles[f"{d['impl']}/r{d['rows']}"] = {
            "aten": d["dispatches_per_forward"],
            "custom": d.get("custom_kernel_launches_per_forward"),
            "total": d.get("launches_per_forward", d["dispatches_per_forward"]),
            "complete": d.get("launch_count_complete", True),
        }
    prov = (
        (bench or serving or correctness)[0].provenance if (bench or serving or correctness) else {}
    )
    comps = [c.to_dict() for c in comparisons(bench)]
    return {
        "schema": 1,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "evidence": {
            "benchmarks": "measured",
            "comparisons": "derived (paired per-trial ratios, bootstrap 95% CI)",
            "roofline.achieved_gbps": "derived",
            "roofline.nominal_gbps / floor_us": "theoretical (vendor-nominal bandwidth)",
            "serving": "measured (open-loop replay)",
        },
        "environment": {
            "os": f"{prov.get('os')} {prov.get('os_release')}",
            "cpu": prov.get("cpu"),
            "memory_gib": round((prov.get("memory_bytes") or 0) / 2**30, 1),
            "torch": prov.get("torch"),
            "python": prov.get("python"),
            "accelerators": prov.get("accelerators"),
            "git_commit": prov.get("git_commit"),
        },
        "benchmarks": [_bench_entry(r) for r in bench],
        "comparisons": comps,
        "regressions": [c for c in comps if c["verdict"] == "slower"],
        "serving": [_serving_entry(r) for r in serving],
        "envelope": _load_json(p4 / "envelope.json"),
        "idle_gap": _load_json(p4 / "idle_gap_interleaved.json"),
        "kernels": {
            "launches_per_forward": profiles,
            "metal_tuning": _load_json(p3 / "metal_tuning.json"),
            "metal_v1_vs_v2": _load_json(p3 / "metal_v1_vs_v2.json"),
        },
        "baseline_phase2": {
            "repeatability": _load_json(p2 / "repeatability_mps.json"),
        },
        "correctness": [
            {
                "device": r.device,
                "dtype": r.dtype,
                "rows": r.rows,
                "impl": r.impl,
                **{k: (r.correctness or {}).get(k) for k in ("rel_l2", "max_scaled", "passed")},
            }
            for r in correctness
        ],
        "unavailable": [
            *unavailable,
            "NVIDIA GPU results (no NVIDIA hardware available or authorized)",
            "Nsight Systems / Nsight Compute traces (not available on macOS)",
        ],
    }


def write_summary(results: Path, out: Path) -> dict[str, Any]:
    s = build_summary(results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, separators=(",", ":"), default=str) + "\n")
    return s
