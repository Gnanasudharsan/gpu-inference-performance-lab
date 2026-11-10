"""Structured, machine-readable experiment records (JSON Lines, Parquet)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpulab.provenance import collect_provenance

SCHEMA_VERSION = 1

#: How a number was obtained. Anything other than "measured" must never be
#: presented as an observation.
EVIDENCE = ("measured", "derived", "theoretical")


@dataclass
class ExperimentRecord:
    kind: str  # "correctness" | "benchmark" | "serving"
    impl: str
    device: str
    device_name: str
    dtype: str
    rows: int
    model: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    correctness: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    evidence: str = "measured"
    provenance: dict[str, Any] = field(default_factory=lambda: collect_provenance().to_dict())
    created_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds")
    )
    schema_version: int = SCHEMA_VERSION
    record_id: str = ""

    def __post_init__(self) -> None:
        if self.evidence not in EVIDENCE:
            raise ValueError(f"evidence must be one of {EVIDENCE}")
        if not self.record_id:
            key = json.dumps(
                [
                    self.kind,
                    self.impl,
                    self.device,
                    self.dtype,
                    self.rows,
                    self.model,
                    self.params,
                    self.created_utc,
                ],
                sort_keys=True,
                default=str,
            )
            self.record_id = f"{self.kind[:3]}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentRecord:
        return cls(**d)


def append_jsonl(path: str | Path, records: Iterable[ExperimentRecord]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[ExperimentRecord]:
    out = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                out.append(ExperimentRecord.from_dict(json.loads(line)))
    return out


def flatten(record: ExperimentRecord) -> dict[str, Any]:
    """One flat row per record, suitable for a DataFrame (lists are JSON-encoded)."""
    d = record.to_dict()
    row: dict[str, Any] = {}

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}{k}." if prefix or k else k, v)
        elif isinstance(obj, list):
            row[prefix.rstrip(".")] = json.dumps(obj, default=str)
        else:
            row[prefix.rstrip(".")] = obj

    for key in ("model", "metrics", "correctness", "params"):
        walk(f"{key}.", d.pop(key) or {})
    prov = d.pop("provenance")
    row["git_commit"] = prov.get("git_commit")
    row["torch"] = prov.get("torch")
    row["os"] = f"{prov.get('os')} {prov.get('os_release')}"
    row.update(d)
    return row


def write_parquet(path: str | Path, records: Iterable[ExperimentRecord]) -> Path:
    import pandas as pd

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([flatten(r) for r in records]).to_parquet(p, index=False)
    return p
