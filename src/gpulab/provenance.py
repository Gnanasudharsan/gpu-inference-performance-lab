"""Hardware and software provenance capture.

Every experiment record embeds the output of :func:`collect_provenance` so that a
number can always be traced back to the exact machine, accelerator, library
versions and Git commit that produced it.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        # fixed argument lists from this module only; no shell, no user input
        out = subprocess.run(  # noqa: S603
            cmd, cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _cpu_brand() -> str:
    if sys.platform == "darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    if name and name.lower() != "unknown":
                        return name
        except OSError:
            pass
    # sandboxes (e.g. gVisor on Modal) may report the model name as "unknown"
    return platform.processor() or f"{platform.machine()} CPU"


def _total_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        mem = _run(["sysctl", "-n", "hw.memsize"])
        return int(mem) if mem and mem.isdigit() else None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (ValueError, OSError, AttributeError):
        return None


def _triton_version() -> str | None:
    try:
        import triton

        return str(triton.__version__)
    except Exception:
        return None


@dataclass(frozen=True)
class AcceleratorInfo:
    backend: str  # "cuda" | "mps" | "cpu"
    name: str
    count: int = 1
    memory_bytes: int | None = None
    compute_capability: str | None = None
    driver: str | None = None


@dataclass(frozen=True)
class Provenance:
    timestamp_utc: str
    hostname_hash: str
    os: str
    os_release: str
    machine: str
    cpu: str
    cpu_count: int | None
    memory_bytes: int | None
    python: str
    torch: str
    torch_cuda: str | None
    triton: str | None
    numpy: str
    accelerators: list[AcceleratorInfo] = field(default_factory=list)
    git_commit: str | None = None
    git_dirty: bool | None = None
    #: tracked files that differ from HEAD (e.g. docs only vs. source changes)
    git_dirty_files: list[str] = field(default_factory=list)
    tools: dict[str, bool] = field(default_factory=dict)
    env_flags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def accelerators() -> list[AcceleratorInfo]:
    accs: list[AcceleratorInfo] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            accs.append(
                AcceleratorInfo(
                    backend="cuda",
                    name=p.name,
                    memory_bytes=int(p.total_memory),
                    compute_capability=f"{p.major}.{p.minor}",
                    driver=_run(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
                    ),
                )
            )
    if torch.backends.mps.is_available():
        # Apple GPUs share unified memory with the CPU; report the chip name.
        accs.append(AcceleratorInfo(backend="mps", name=f"{_cpu_brand()} GPU (Metal/MPS)"))
    accs.append(AcceleratorInfo(backend="cpu", name=_cpu_brand(), count=os.cpu_count() or 1))
    return accs


def git_state(root: Path = REPO_ROOT) -> tuple[str | None, bool | None, list[str]]:
    """HEAD commit, whether tracked files differ from it, and which ones."""
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if commit is None:
        # deployed containers have no .git: the deploy step passes the exact commit instead
        return os.environ.get("GPULAB_GIT_COMMIT") or None, None, []
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root) or ""
    # porcelain lines are "XY path" (or "XY old -> new"); _run strips the first line's
    # leading space, so split on whitespace instead of slicing fixed columns.
    files = sorted(
        line.split(None, 1)[1].split(" -> ")[-1] for line in status.splitlines() if line.strip()
    )
    return commit, bool(files), files


@lru_cache(maxsize=1)
def _static_provenance() -> dict[str, Any]:
    import hashlib

    import numpy as np

    return {
        "hostname_hash": hashlib.sha256(platform.node().encode()).hexdigest()[:12],
        "os": platform.system(),
        "os_release": platform.mac_ver()[0] or platform.release(),
        "machine": platform.machine(),
        "cpu": _cpu_brand(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _total_memory_bytes(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": _triton_version(),
        "numpy": np.__version__,
        "tools": {t: shutil.which(t) is not None for t in ("nvidia-smi", "nvcc", "nsys", "ncu")},
    }


def collect_provenance() -> Provenance:
    """Collect a provenance snapshot. Hostnames are hashed, never stored raw."""
    commit, dirty, dirty_files = git_state()
    flags = {
        k: v
        for k, v in os.environ.items()
        if k
        in {
            "TRITON_INTERPRET",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "PYTORCH_ENABLE_MPS_FALLBACK",
            "CUBLAS_WORKSPACE_CONFIG",
        }
    }
    return Provenance(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        accelerators=accelerators(),
        git_commit=commit,
        git_dirty=dirty,
        git_dirty_files=dirty_files,
        env_flags=flags,
        **_static_provenance(),
    )


def format_report(p: Provenance) -> str:
    gib = (p.memory_bytes or 0) / 2**30
    lines = [
        f"timestamp (UTC) : {p.timestamp_utc}",
        f"OS              : {p.os} {p.os_release} ({p.machine})",
        f"CPU             : {p.cpu} x{p.cpu_count}",
        f"memory          : {gib:.1f} GiB",
        f"python          : {p.python}",
        f"torch           : {p.torch} (CUDA build: {p.torch_cuda or 'none'})",
        f"triton          : {p.triton or 'not installed'}",
        f"numpy           : {p.numpy}",
        f"git             : {p.git_commit or 'n/a'}{' (dirty)' if p.git_dirty else ''}",
        "accelerators    :",
    ]
    for a in p.accelerators:
        extra = []
        if a.memory_bytes:
            extra.append(f"{a.memory_bytes / 2**30:.1f} GiB")
        if a.compute_capability:
            extra.append(f"sm_{a.compute_capability.replace('.', '')}")
        if a.driver:
            extra.append(f"driver {a.driver}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"  - [{a.backend}] {a.name}{suffix}")
    lines.append("profiling tools :")
    for tool, ok in p.tools.items():
        lines.append(f"  - {tool:<10} {'found' if ok else 'not found'}")
    return "\n".join(lines)


def system_state() -> dict[str, Any]:
    """Per-run conditions that change performance: power source, thermal/clock state.

    macOS: ``pmset`` power source and thermal/performance warning levels (no root needed).
    NVIDIA: SM/memory clocks, temperature and power draw from ``nvidia-smi``.
    """
    state: dict[str, Any] = {}
    if sys.platform == "darwin":
        batt = _run(["pmset", "-g", "batt"]) or ""
        state["power_source"] = (
            "AC" if "AC Power" in batt else "battery" if "Battery Power" in batt else None
        )
        therm = _run(["pmset", "-g", "therm"]) or ""
        state["thermal_warning"] = "No thermal warning level has been recorded" not in therm
        state["performance_warning"] = "No performance warning level has been recorded" not in therm
        low_power = _run(["pmset", "-g"]) or ""
        for line in low_power.splitlines():
            if line.strip().startswith("lowpowermode"):
                state["low_power_mode"] = line.split()[-1] == "1"
    if shutil.which("nvidia-smi"):
        q = _run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate",
                "--format=csv,noheader,nounits",
            ]
        )
        if q:
            sm, mem, temp, power, pstate = (v.strip() for v in q.splitlines()[0].split(","))
            state.update(
                {
                    "sm_clock_mhz": sm,
                    "mem_clock_mhz": mem,
                    "gpu_temp_c": temp,
                    "power_w": power,
                    "pstate": pstate,
                }
            )
    return state
