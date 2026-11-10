"""Command-line entry point: ``gpulab <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any


def _cmd_env(args: argparse.Namespace) -> int:
    from gpulab.provenance import collect_provenance, format_report

    prov = collect_provenance()
    print(json.dumps(prov.to_dict(), indent=2) if args.json else format_report(prov))
    return 0


def add_workload_args(p: argparse.ArgumentParser, multi: bool = False) -> None:
    g = p.add_argument_group("workload")
    if multi:
        g.add_argument("--impls", nargs="+", default=["eager"], help="execution modes")
        g.add_argument("--rows", nargs="+", type=int, default=[1, 8, 32], help="batch sizes")
    else:
        g.add_argument("--impl", default="eager", help="execution mode (see `gpulab impls`)")
        g.add_argument("--rows", type=int, default=8, help="batch size (rows per request)")
    g.add_argument("--device", default="auto", help="cpu | mps | cuda | auto")
    g.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    g.add_argument("--d-model", type=int, default=1024)
    g.add_argument("--d-hidden", type=int, default=2816)
    g.add_argument("--depth", type=int, default=4)
    g.add_argument("--seed", type=int, default=0, help="weight seed")
    g.add_argument("--input-seed", type=int, default=1234)


def _cmd_impls(args: argparse.Namespace) -> int:
    import torch

    from gpulab import impls
    from gpulab.device import available_devices, parse_dtype

    for name in impls.names():
        impl = impls.get(name)
        support = []
        for dev in available_devices():
            for dt in ("float32", "float16", "bfloat16"):
                reason = impl.unsupported_reason(torch.device(dev), parse_dtype(dt))
                support.append(f"{dev}/{dt}:{'yes' if reason is None else 'no'}")
        print(f"{name:<16} {impl.description}")
        print(f"{'':<16} {' '.join(support)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from gpulab import impls
    from gpulab.correctness import compare
    from gpulab.device import device_name, parse_dtype, resolve_device, synchronize
    from gpulab.records import ExperimentRecord, append_jsonl
    from gpulab.workload import ModelConfig, describe, make_inputs, make_weights, reference_forward

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    cfg = ModelConfig(
        d_model=args.d_model, d_hidden=args.d_hidden, depth=args.depth, seed=args.seed
    )
    impl = impls.get(args.impl)
    reason = impl.unsupported_reason(device, dtype)
    if reason:
        print(f"{impl.name} unsupported on {device.type}/{args.dtype}: {reason}", file=sys.stderr)
        return 2
    weights = make_weights(cfg, device, dtype)
    x = make_inputs(args.rows, cfg.d_model, args.input_seed, device, dtype)
    runner = impl.build(weights, rows=args.rows)
    y = runner(x)
    synchronize(device)
    report = compare(y, reference_forward(weights, x), dtype)
    print(
        f"impl={impl.name} device={device.type} ({device_name(device)}) dtype={args.dtype}"
        f" rows={args.rows} model={cfg.d_model}x{cfg.d_hidden}x{cfg.depth}"
    )
    print(report)
    if args.save:
        rec = ExperimentRecord(
            kind="correctness",
            impl=impl.name,
            device=device.type,
            device_name=device_name(device),
            dtype=args.dtype,
            rows=args.rows,
            model=cfg.to_dict(),
            correctness=report.to_dict(),
            params={"input_seed": args.input_seed, **describe(cfg, args.rows, dtype)},
        )
        append_jsonl(args.save, [rec])
        print(f"saved {rec.record_id} -> {args.save}")
    return 0 if report.passed else 1


def _model_from(args: argparse.Namespace) -> Any:
    from gpulab.workload import ModelConfig

    return ModelConfig(
        d_model=args.d_model, d_hidden=args.d_hidden, depth=args.depth, seed=args.seed
    )


def _cmd_bench(args: argparse.Namespace) -> int:
    from gpulab.bench import BenchConfig, run_benchmark, run_interleaved
    from gpulab.device import parse_dtype, resolve_device
    from gpulab.records import append_jsonl
    from gpulab.report import benchmark_table

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    model = _model_from(args)
    cfg = BenchConfig(
        iters=args.iters,
        trials=args.trials,
        warmup_s=args.warmup_s,
        idle_gap_us=args.idle_gap_us,
        deadline_us=args.deadline_us,
        device_iters=args.device_iters,
        keep_samples=not args.no_samples,
    )
    records = []
    for rows in args.rows:
        if args.interleave:
            records += run_interleaved(
                list(args.impls), model, rows, device, dtype, cfg, args.input_seed
            )
        else:
            for name in args.impls:
                records += run_benchmark(
                    name, model, rows, device, dtype, cfg, tuple(args.modes), args.input_seed
                )
    print(benchmark_table(records, baseline=args.impls[0]))
    if args.save:
        append_jsonl(args.save, records)
        print(f"saved {len(records)} records -> {args.save}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    from gpulab.device import parse_dtype, resolve_device
    from gpulab.profiling import profile_forward

    out = profile_forward(
        args.impl,
        _model_from(args),
        args.rows,
        resolve_device(args.device),
        parse_dtype(args.dtype),
        iters=args.iters,
        trace_path=args.trace,
    )
    print(json.dumps(out, indent=2))
    if args.save:
        from pathlib import Path

        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(out, indent=2) + "\n")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from gpulab.records import read_jsonl, write_parquet
    from gpulab.report import benchmark_table

    records = [r for path in args.inputs for r in read_jsonl(path)]
    table = benchmark_table(records, baseline=args.baseline)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(table + "\n")
    else:
        print(table)
    if args.parquet:
        write_parquet(args.parquet, records)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpulab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="print hardware/software provenance")
    env.add_argument("--json", action="store_true", help="emit JSON")
    env.set_defaults(func=_cmd_env)

    lst = sub.add_parser("impls", help="list execution modes and where they are supported")
    lst.set_defaults(func=_cmd_impls)

    run = sub.add_parser("run", help="run one forward pass and check it against the reference")
    add_workload_args(run)
    run.add_argument("--save", help="append a correctness record to this JSONL file")
    run.set_defaults(func=_cmd_run)

    bench = sub.add_parser("bench", help="benchmark one or more implementations")
    add_workload_args(bench, multi=True)
    bench.add_argument(
        "--modes", nargs="+", default=["latency"], choices=["latency", "e2e", "throughput"]
    )
    bench.add_argument(
        "--interleave",
        action="store_true",
        help="interleave impls trial-by-trial (latency mode, paired speedups)",
    )
    bench.add_argument("--iters", type=int, default=300)
    bench.add_argument("--trials", type=int, default=3)
    bench.add_argument("--burst", type=int, default=50, help="requests per pipelined burst")
    bench.add_argument("--warmup-s", type=float, default=1.0)
    bench.add_argument("--idle-gap-us", type=float, default=0.0)
    bench.add_argument("--device-iters", type=int, default=100)
    bench.add_argument("--deadline-us", type=float, default=None)
    bench.add_argument("--no-samples", action="store_true", help="do not store raw samples")
    bench.add_argument("--save", help="append records to this JSONL file")
    bench.set_defaults(func=_cmd_bench)

    prof = sub.add_parser("profile", help="PyTorch-profiler summary of one forward pass")
    add_workload_args(prof)
    prof.add_argument("--iters", type=int, default=20)
    prof.add_argument("--trace", help="write a Chrome trace (JSON) here")
    prof.add_argument("--save", help="write the summary JSON here")
    prof.set_defaults(func=_cmd_profile)

    rep = sub.add_parser("report", help="markdown table from benchmark JSONL files")
    rep.add_argument("inputs", nargs="+")
    rep.add_argument("--baseline", default="eager")
    rep.add_argument("--out", help="write markdown here")
    rep.add_argument("--parquet", help="also write a flattened Parquet file")
    rep.set_defaults(func=_cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
