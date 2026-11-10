"""Modal deployment of the gpulab execution API on one NVIDIA GPU (public demo).

STATUS: prepared, NOT deployed. Deploying requires the owner's Modal account (free Starter plan
includes monthly credits) and an agreed spending cap (see docs/DEPLOYMENT.md).

Cost and abuse controls encoded here:
  * gpu="L4", max_containers=1      -> at most one GPU billed at any moment
  * @modal.concurrent(max_inputs=8) -> health checks answer while a measurement runs; the app
                                       itself still runs ONE measurement at a time (429 if busy)
  * timeout=120                     -> hard per-request wall-clock limit
  * scaledown_window=60             -> scale to zero one minute after the last request
  * GPULAB_DAILY_RUN_CAP=8          -> global measurements per UTC day (the spend bound)
  * GPULAB_RATE_PER_MIN=3           -> per-client rate limit
Worst case with these settings: 8 runs/day, each waking the GPU for ~2.5 min (cold start +
run + 60 s scale-down) -> ~20 GPU-min/day -> ~10 GPU-h/month, about $8/month at the reported
~$0.80/h L4 rate. Deployed on an account WITHOUT a payment method, nothing can be charged:
when the free credit runs out the endpoint simply stops answering until it renews (the
dashboard's Live page then reports the backend offline).

Deploy (from the repository root):
    modal secret create gpulab-api GPULAB_CORS_ORIGINS=https://gpu-inference-performance-lab.vercel.app
    PYTHONPATH=src modal deploy deploy/modal_app.py
"""

import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
    .pip_install("fastapi>=0.110", "uvicorn>=0.29", "pydantic>=2.6", "triton>=3.1", "numpy>=1.26")
    .env({"GPULAB_DEVICE": "cuda", "GPULAB_TRUST_PROXY": "1", "GPULAB_DAILY_RUN_CAP": "8",
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
