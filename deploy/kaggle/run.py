"""Kaggle kernel entry point: measure gpulab on Kaggle's free NVIDIA T4.

Pushed by scripts/kaggle_run.sh, which pins COMMIT to the exact commit being measured, so the
results are traceable to the code that produced them. Outputs land in /kaggle/working.
"""

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/Gnanasudharsan/gpu-inference-performance-lab"
COMMIT = "__COMMIT__"
CONFIG = "experiments/nvidia_t4.toml"
SRC = Path("/tmp/gpulab")  # not /kaggle/working: keep the source out of the outputs
OUT = Path("/kaggle/working/gpulab_results")


def sh(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


SRC.parent.mkdir(parents=True, exist_ok=True)
sh("git", "clone", "--filter=blob:none", REPO, str(SRC))
sh("git", "checkout", "--detach", COMMIT, cwd=SRC)
sh("nvidia-smi")
# keep Kaggle's preinstalled CUDA build of torch (and its bundled triton); add only our deps
sh(sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]", cwd=SRC)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "pip_freeze.txt").write_text(
    subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
)
rc = subprocess.run(
    [sys.executable, "scripts/gpu_session.py", "--config", CONFIG, "--out", str(OUT)], cwd=SRC
).returncode
print(f"gpu_session exit code: {rc}", flush=True)
