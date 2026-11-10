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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpulab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="print hardware/software provenance")
    env.add_argument("--json", action="store_true", help="emit JSON")
    env.set_defaults(func=_cmd_env)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
