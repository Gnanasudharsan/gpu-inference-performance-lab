# GPU execution service image (NVIDIA). NOT BUILT in the development environment: the Docker
# daemon was unavailable and disk space was limited. Build and test on a CUDA host:
#   docker build -f deploy/Dockerfile -t gpulab-api .
#   docker run --gpus all -p 8000:8000 -e GPULAB_API_KEY=... -e GPULAB_CORS_ORIGINS=https://gpu-inference-performance-lab.vercel.app gpulab-api
ARG BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GPULAB_DEVICE=auto GPULAB_RATE_PER_MIN=6 GPULAB_MAX_RUN_S=30

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[serve,triton]" && python -c "import gpulab, fastapi, uvicorn"

# least privilege: no root at runtime, read-only app directory
RUN useradd --create-home --uid 10001 gpulab && chown -R root:root /app && chmod -R a-w /app
USER gpulab

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "gpulab.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
