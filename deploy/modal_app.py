"""Modal deployment of the gpulab execution API (public demo).

Two modes, chosen at deploy time with GPULAB_MODAL_GPU:

  * "none" (default): CPU container. Works on a free Modal account WITHOUT a payment method
    (Modal requires a payment method for GPU functions: "Please add a payment method to use
    L4 GPU functions", observed 2026-11-10). Live runs compare the CPU implementations.
  * "L4" (or another GPU type): one NVIDIA GPU; needs a payment method on the account.

Cost and abuse controls (both modes):
  * max_containers=1                -> at most one container billed at any moment
  * @modal.concurrent(max_inputs=8) -> health checks answer while a measurement runs; the app
                                       itself runs ONE measurement at a time (429 if busy)
  * timeout=120, scaledown_window=60 -> hard per-request limit; scale to zero after 1 min idle
  * GPULAB_DAILY_RUN_CAP            -> global measurements per UTC day (8 on GPU, 40 on CPU)
  * GPULAB_RATE_PER_MIN=3           -> per-client rate limit
On an account without a payment method nothing can be charged: if the free credit runs out the
endpoint stops answering and the dashboard's Live page reports the backend offline.

Deploy (from the repository root):
    modal secret create gpulab-api GPULAB_CORS_ORIGINS=https://gpu-inference-performance-lab.vercel.app
    PYTHONPATH=src modal deploy deploy/modal_app.py                       # CPU (default)
    GPULAB_MODAL_GPU=L4 PYTHONPATH=src modal deploy deploy/modal_app.py   # GPU (needs a card)
"""

import os
import subprocess

import modal

# the container has no .git: pass the exact deployed commit so live results stay traceable
try:
    COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
except (OSError, subprocess.CalledProcessError):
    COMMIT = ""

GPU = os.environ.get("GPULAB_MODAL_GPU", "none")
ON_GPU = GPU.lower() != "none"
API_DEPS = ("fastapi>=0.110", "uvicorn>=0.29", "pydantic>=2.6", "numpy>=1.26")

if ON_GPU:
    image = modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime").pip_install(
        *API_DEPS, "triton>=3.1"
    )
else:
    image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu"
    ).pip_install(*API_DEPS)

image = image.env(
    {
        "GPULAB_DEVICE": "cuda" if ON_GPU else "cpu",
        "GPULAB_TRUST_PROXY": "1",
        "GPULAB_DAILY_RUN_CAP": "8" if ON_GPU else "40",
        "GPULAB_RATE_PER_MIN": "3",
        "GPULAB_HOST_LABEL": f"Modal cloud GPU ({GPU})" if ON_GPU else "Modal cloud CPU (2 vCPU)",
        "GPULAB_GIT_COMMIT": COMMIT,
    }
).add_local_python_source("gpulab")  # resolved from src/ via PYTHONPATH=src at deploy time

app = modal.App("gpulab-api", image=image)

# persistent counter for the daily cap: survives scale-to-zero restarts (in-memory would not)
usage = modal.Dict.from_name("gpulab-api-usage", create_if_missing=True)


@app.function(
    gpu=GPU if ON_GPU else None,
    cpu=2.0,
    memory=4096,
    max_containers=1,
    timeout=120,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("gpulab-api")],
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def api():  # type: ignore[no-untyped-def]
    from gpulab.api import create_app

    return create_app(usage_store=usage)
