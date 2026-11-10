"""One-shot measurement session for a remote NVIDIA GPU (Kaggle, a cluster node, a cloud VM).

Steps, each recorded in <out>/session.json with status ok | failed | skipped and a reason:

  1. provenance       gpulab env --json, nvidia-smi -q
  2. correctness gate the full test suite on this machine (CUDA + Triton tests included).
                      If it fails, NO benchmarks are run (results would be meaningless).
  3. experiment       gpulab experiment <config>  -> <out>/experiment/{bench,serving}.jsonl
  4. kernel profiles  PyTorch profiler with CUDA activity: kernels and kernel time per forward
  5. nsys (optional)  Nsight Systems timelines for eager vs CUDA-graph replay, if nsys exists
  6. archive          <out>.tar.gz for download

Usage:
  python scripts/gpu_session.py --config experiments/nvidia_t4.toml --out /kaggle/working/gpulab
  python scripts/gpu_session.py --config experiments/smoke.toml --out /tmp/s --tests tests/x.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MIN_CAPABILITY = (7, 5)  # Triton kernels and fp16 tensor cores: Turing (T4) or newer


class Session:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.steps: list[dict[str, Any]] = []
        out.mkdir(parents=True, exist_ok=True)
        self.log = (out / "session.log").open("a")

    def say(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def step(self, name: str, fn: Callable[[], Any], required: bool = False) -> bool:
        self.say(f"== {name}")
        t0 = time.time()
        rec: dict[str, Any] = {"step": name}
        try:
            detail = fn()
            rec.update(status="ok", detail=detail)
        except SkipStepError as s:
            rec.update(status="skipped", reason=str(s))
            self.say(f"   skipped: {s}")
        except Exception as e:
            rec.update(
                status="failed",
                reason=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc()[-4000:],
            )
            self.say(f"   FAILED: {type(e).__name__}: {e}")
        rec["seconds"] = round(time.time() - t0, 1)
        self.steps.append(rec)
        self.save()
        # a skipped step (e.g. no CUDA on a dry run) does not abort; a failed required step does
        return rec["status"] in ("ok", "skipped") or not required

    def save(self) -> None:
        (self.out / "session.json").write_text(
            json.dumps({"steps": self.steps}, indent=2, default=str) + "\n"
        )


class SkipStepError(Exception):
    pass


def run(cmd: list[str], log: Path, timeout: float | None = None) -> int:
    with log.open("w") as f:
        return subprocess.run(
            cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=False
        ).returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/nvidia_t4.toml")
    p.add_argument("--out", required=True)
    p.add_argument("--tests", nargs="*", default=["tests"], help="pytest targets for the gate")
    p.add_argument("--profile-rows", nargs="+", type=int, default=[8, 32])
    a = p.parse_args()
    out = Path(a.out).resolve()
    s = Session(out)

    import torch

    def provenance() -> dict[str, Any]:
        from gpulab.provenance import collect_provenance

        prov = collect_provenance().to_dict()
        (out / "env.json").write_text(json.dumps(prov, indent=2) + "\n")
        if shutil.which("nvidia-smi"):
            run(["nvidia-smi", "-q"], out / "nvidia_smi.txt")
        cuda = torch.cuda.is_available()
        info: dict[str, Any] = {"cuda": cuda, "torch": torch.__version__, "triton": prov["triton"]}
        if cuda:
            cap = torch.cuda.get_device_capability()
            info.update(gpu=torch.cuda.get_device_name(), capability=f"{cap[0]}.{cap[1]}")
        return info

    s.step("provenance", provenance, required=True)

    def capability_gate() -> str:
        if not torch.cuda.is_available():
            raise SkipStepError("no CUDA device: CUDA-specific steps will be skipped")
        cap = torch.cuda.get_device_capability()
        if cap < MIN_CAPABILITY:
            raise RuntimeError(
                f"compute capability {cap} < {MIN_CAPABILITY}: Triton paths "
                "unsupported here (request a T4 or newer)"
            )
        return f"capability {cap} ok"

    if not s.step("capability gate", capability_gate, required=True):
        s.say("aborting: unsupported GPU")
        return finish(s, out, 1)

    def correctness_gate() -> str:
        junit = out / "pytest.xml"
        rc = run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={junit}",
                *a.tests,
            ],
            out / "pytest.log",
            timeout=3600,
        )
        tail = (out / "pytest.log").read_text().strip().splitlines()[-1:]
        if rc != 0:
            raise RuntimeError(f"pytest exit {rc}: {tail}")
        return tail[0] if tail else "passed"

    if not s.step("correctness gate", correctness_gate, required=True):
        s.say("aborting: correctness gate failed, no benchmarks were run")
        return finish(s, out, 1)

    def experiment() -> dict[str, Any]:
        from gpulab.experiment import load_experiment, run_experiment

        manifest = run_experiment(load_experiment(ROOT / a.config), out / "experiment", log=s.say)
        return {
            "counts": manifest["counts"],
            "skipped": manifest["skipped"],
            "elapsed_s": manifest["elapsed_s"],
        }

    s.step("experiment", experiment)

    def profiles() -> dict[str, Any]:
        if not torch.cuda.is_available():
            raise SkipStepError("no CUDA device")
        from gpulab import impls
        from gpulab.profiling import profile_forward
        from gpulab.workload import ModelConfig

        done = {}
        dev = torch.device("cuda")
        for name in (
            "eager",
            "fused_weights",
            "compile",
            "triton",
            "cuda_graph_triton",
            "compile_cudagraphs",
        ):
            if not impls.get(name).supports(dev, torch.float16):
                continue
            for rows in a.profile_rows:
                r = profile_forward(name, ModelConfig(), rows, dev, torch.float16, iters=20)
                path = out / f"profile_{name}_r{rows}_cuda.json"
                path.write_text(json.dumps(r, indent=2) + "\n")
                done[f"{name}/r{rows}"] = {
                    "cuda_kernels_per_forward": r.get("cuda_kernels_per_forward"),
                    "cuda_kernel_us_per_forward": r.get("cuda_kernel_us_per_forward"),
                }
        return done

    s.step("kernel profiles", profiles)

    def nsys() -> list[str]:
        if not shutil.which("nsys"):
            raise SkipStepError("nsys not installed on this machine")
        made = []
        for impl in ("eager", "cuda_graph_triton"):
            rc = run(
                ["bash", "scripts/profile_nsys.sh", impl, "8"],
                out / f"nsys_{impl}.log",
                timeout=1800,
            )
            if rc != 0:
                raise RuntimeError(f"profile_nsys.sh {impl} exit {rc}")
            made.append(impl)
        for f in (ROOT / "results/gpu/nsys").glob("*.csv"):
            shutil.copy(f, out / f.name)
        return made

    s.step("nsight systems", nsys)
    return finish(s, out, 0)


def finish(s: Session, out: Path, rc: int) -> int:
    s.save()
    s.log.close()
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as t:
        t.add(out, arcname=out.name)
    print(f"archive: {archive}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
