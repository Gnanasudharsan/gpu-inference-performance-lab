"""Modal deployment of the gpulab execution API on one NVIDIA GPU.

STATUS: prepared, NOT deployed. Deploying requires a Modal account, an agreed spending cap
and the owner's authorization (see docs/DEPLOYMENT.md).

Cost and abuse controls encoded here:
  * gpu="L4", max_containers=1      -> at most one GPU billed at any moment
  * @modal.concurrent(max_inputs=1) -> one measurement at a time (also enforced in the app)
  * timeout=120                     -> hard per-request wall-clock limit
  * scaledown_window=120            -> scale to zero two minutes after the last request
  * API key from a Modal Secret     -> never in code or the image
Set a workspace spending limit in the Modal dashboard as the outer cap.

Deploy (from the repository root):
    modal secret create gpulab-api GPULAB_API_KEY=<random> \
        GPULAB_CORS_ORIGINS=https://gpu-inference-performance-lab.vercel.app
    modal deploy deploy/modal_app.py
"""

import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
    .pip_install("fastapi>=0.110", "uvicorn>=0.29", "pydantic>=2.6", "triton>=3.1", "numpy>=1.26")
    .add_local_python_source("gpulab")  # resolved from src/ via PYTHONPATH=src at deploy time
)

app = modal.App("gpulab-api", image=image)


@app.function(
    gpu="L4",
    max_containers=1,
    timeout=120,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("gpulab-api")],
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def api():  # type: ignore[no-untyped-def]
    import os

    os.environ.setdefault("GPULAB_DEVICE", "cuda")
    os.environ.setdefault("GPULAB_TRUST_PROXY", "1")  # client address from X-Forwarded-For
    from gpulab.api import create_app

    return create_app()
