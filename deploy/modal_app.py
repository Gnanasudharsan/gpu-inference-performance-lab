"""Modal deployment of the gpulab execution API on one NVIDIA GPU (public demo).

STATUS: prepared, NOT deployed. Deploying requires the owner's Modal account (free Starter plan
includes monthly credits) and an agreed spending cap (see docs/DEPLOYMENT.md).

Cost and abuse controls encoded here:
  * gpu="L4", max_containers=1      -> at most one GPU billed at any moment
  * @modal.concurrent(max_inputs=8) -> health checks answer while a measurement runs; the app
                                       itself still runs ONE measurement at a time (429 if busy)
  * timeout=120                     -> hard per-request wall-clock limit
  * scaledown_window=60             -> scale to zero one minute after the last request
  * GPULAB_DAILY_RUN_CAP=30         -> global measurements per UTC day (the spend bound)
  * GPULAB_RATE_PER_MIN=3           -> per-client rate limit
Worst case with these settings: 30 runs/day, each waking the GPU for <= ~3 min (cold start +
run + scale-down) -> <= 1.5 GPU-h/day. Set a workspace spending limit in the Modal dashboard as
the outer cap.

Deploy (from the repository root):
    modal secret create gpulab-api GPULAB_CORS_ORIGINS=https://gpu-inference-performance-lab.vercel.app
    PYTHONPATH=src modal deploy deploy/modal_app.py
"""

import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
    .pip_install("fastapi>=0.110", "uvicorn>=0.29", "pydantic>=2.6", "triton>=3.1", "numpy>=1.26")
    .env({"GPULAB_DEVICE": "cuda", "GPULAB_TRUST_PROXY": "1", "GPULAB_DAILY_RUN_CAP": "30",
          "GPULAB_RATE_PER_MIN": "3"})
    .add_local_python_source("gpulab")  # resolved from src/ via PYTHONPATH=src at deploy time
)

app = modal.App("gpulab-api", image=image)


@app.function(
    gpu="L4",
    max_containers=1,
    timeout=120,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("gpulab-api")],
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def api():  # type: ignore[no-untyped-def]
    from gpulab.api import create_app

    return create_app()
