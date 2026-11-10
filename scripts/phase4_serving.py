"""Phase 4 serving experiments: open-loop rate sweep, bursty arrivals, operating envelope.

For each implementation one server is built and calibrated, then reused for every policy
and arrival rate. Each (policy, rate) replays a deterministic Poisson trace.

Usage: python scripts/phase4_serving.py [--device mps] [--duration 4] [--out results/phase4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gpulab.device import parse_dtype, resolve_device
from gpulab.records import append_jsonl
from gpulab.serving.replay import run_serving_experiment
from gpulab.serving.server import Server, ServerConfig
from gpulab.serving.trace import TraceSpec
from gpulab.workload import ModelConfig, make_weights

POLICIES = {
    "direct": ServerConfig(policy="direct"),
    "static8": ServerConfig(policy="static", batch_rows=8, max_batch_rows=32),
    "dynamic": ServerConfig(policy="dynamic", max_wait_us=500, max_batch_rows=32),
}


def envelope(rows: list[dict], miss_budget: float) -> dict[str, float | None]:
    """Highest offered rate at which the deadline-miss rate stayed within budget."""
    out: dict[str, float | None] = {}
    for key in sorted({r["key"] for r in rows}):
        ok = [r["rate"] for r in rows if r["key"] == key and r["miss"] <= miss_budget]
        out[key] = max(ok) if ok else None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--impls", nargs="+", default=["eager", "metal", "compile"])
    p.add_argument(
        "--rates", nargs="+", type=float, default=[50, 100, 200, 300, 400, 600, 800, 1200, 1600]
    )
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--deadline-us", type=float, default=10000.0)
    p.add_argument("--miss-budget", type=float, default=0.01)
    p.add_argument("--out", default="results/phase4")
    a = p.parse_args()
    device, dtype = resolve_device(a.device), parse_dtype(a.dtype)
    model = ModelConfig()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sweep_path = out / "serving_sweep.jsonl"
    sweep_path.unlink(missing_ok=True)
    rows: list[dict] = []
    for impl in a.impls:
        server = Server(impl, make_weights(model, device, dtype), POLICIES["dynamic"])
        print(
            f"{impl}: calibrated service time per bucket (us):",
            {k: round(v) for k, v in server.calibration.items()},
            flush=True,
        )
        for pname, pcfg in POLICIES.items():
            for rate in a.rates:
                spec = TraceSpec(
                    kind="poisson",
                    rate_rps=rate,
                    duration_s=a.duration,
                    seed=int(rate),
                    deadline_us=a.deadline_us,
                )
                rec, _ = run_serving_experiment(impl, model, device, dtype, pcfg, spec, server)
                rec.params["policy_name"] = pname
                append_jsonl(sweep_path, [rec])
                m = rec.metrics
                lat = m["latency_us"] or {}
                rows.append(
                    {"key": f"{impl}/{pname}", "rate": rate, "miss": m["deadline_miss_rate"]}
                )
                print(
                    f"  {pname:<8} {rate:>6.0f} rps  p50 {lat.get('p50', 0):7.0f}  "
                    f"p99 {lat.get('p99', 0):7.0f} us  miss {m['deadline_miss_rate']:.3f}  "
                    f"goodput {m['goodput_rows_per_s']:7.0f} rows/s  "
                    f"batch {m['mean_batch_rows'] or 0:4.1f}  util {m['utilization']:.2f}",
                    flush=True,
                )
            # bursty arrivals at a moderate mean rate, same policy
            spec = TraceSpec(
                kind="bursty",
                rate_rps=300,
                duration_s=a.duration,
                seed=7,
                deadline_us=a.deadline_us,
            )
            rec, _ = run_serving_experiment(impl, model, device, dtype, pcfg, spec, server)
            rec.params["policy_name"] = pname
            append_jsonl(out / "serving_bursty.jsonl", [rec])
        del server
        if device.type == "mps":
            torch.mps.empty_cache()
    env = envelope(rows, a.miss_budget)
    summary = {
        "miss_budget": a.miss_budget,
        "deadline_us": a.deadline_us,
        "rates": a.rates,
        "max_sustainable_rps": env,
    }
    (out / "envelope.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
