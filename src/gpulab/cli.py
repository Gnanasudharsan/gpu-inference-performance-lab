"""Command-line entry point: ``gpulab <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


def _cmd_env(args: argparse.Namespace) -> int:
    from gpulab.provenance import collect_provenance, format_report

    prov = collect_provenance()
    print(json.dumps(prov.to_dict(), indent=2) if args.json else format_report(prov))
    return 0


def add_workload_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("workload")
    g.add_argument("--impl", default="eager", help="execution mode (see `gpulab impls`)")
    g.add_argument("--device", default="auto", help="cpu | mps | cuda | auto")
    g.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    g.add_argument("--rows", type=int, default=8, help="batch size (rows per request)")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
