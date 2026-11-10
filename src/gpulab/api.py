"""HTTP execution service: bounded, authenticated, rate-limited live benchmarks.

``POST /api/run`` measures one optimized implementation against eager *right now* on this
server's device (interleaved, paired) and checks correctness against the float64
reference. Inputs are enumerated and bounded, so no request can run arbitrary code or
unbounded work.

Configuration (environment variables):

  GPULAB_API_KEY        if set, every /api/run call must send ``X-API-Key`` (constant-time check)
  GPULAB_CORS_ORIGINS   comma-separated allowed browser origins (default: none)
  GPULAB_RATE_PER_MIN   runs per client per minute (default 6)
  GPULAB_DEVICE         cpu | mps | cuda | auto (default auto)
  GPULAB_TRUST_PROXY    "1" to take the client address from X-Forwarded-For (behind a proxy)
  GPULAB_MAX_RUN_S      soft per-request runtime cap reported in /api/health (default 30)
  GPULAB_DAILY_RUN_CAP  global measurements per UTC day (unset: unlimited); bounds demo spend

Run: ``uvicorn gpulab.api:app --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal

import torch
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gpulab import __version__, impls
from gpulab.analysis import paired_ratio_ci, verdict
from gpulab.bench import BenchConfig, run_interleaved
from gpulab.device import device_name, parse_dtype, resolve_device
from gpulab.provenance import collect_provenance
from gpulab.workload import ModelConfig

MODELS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(d_model=64, d_hidden=160, depth=2),
    "small": ModelConfig(d_model=512, d_hidden=1408, depth=4),
    "default": ModelConfig(d_model=1024, d_hidden=2816, depth=4),
}
CANDIDATES = [n for n in impls.names() if n != "eager"]

log = logging.getLogger("gpulab.api")


def _json_log(**fields: Any) -> None:
    fields.setdefault("ts", datetime.now(UTC).isoformat(timespec="milliseconds"))
    log.info(json.dumps(fields, default=str))


class RunRequest(BaseModel):
    impl: str = Field("metal", description="optimized implementation to compare with eager")
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"
    rows: int = Field(8, ge=1, le=64)
    model: Literal["tiny", "small", "default"] = "default"
    mode: Literal["throughput", "latency"] = "throughput"
    trials: int = Field(10, ge=3, le=20)

    model_config = {"extra": "forbid"}


class RateLimiter:
    """Sliding-window limiter: at most ``limit`` events per ``window_s`` per key."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit, self.window = limit, window_s
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        t = time.monotonic() if now is None else now
        with self.lock:
            q = self.events[key]
            while q and t - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, self.window - (t - q[0])
            q.append(t)
            return True, 0.0


class DailyCap:
    """Global cap on measurements per UTC day: bounds total GPU spend for a public demo."""

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.day = datetime.now(UTC).date()
        self.used = 0
        self.lock = threading.Lock()

    def _roll(self) -> None:
        today = datetime.now(UTC).date()
        if today != self.day:
            self.day, self.used = today, 0

    def remaining(self) -> int | None:
        with self.lock:
            self._roll()
            return None if self.limit is None else max(0, self.limit - self.used)

    def take(self) -> bool:
        with self.lock:
            self._roll()
            if self.limit is not None and self.used >= self.limit:
                return False
            self.used += 1
            return True


def create_app() -> FastAPI:
    api_key = os.environ.get("GPULAB_API_KEY") or None
    origins = [o.strip() for o in os.environ.get("GPULAB_CORS_ORIGINS", "").split(",") if o.strip()]
    trust_proxy = os.environ.get("GPULAB_TRUST_PROXY") == "1"
    max_run_s = float(os.environ.get("GPULAB_MAX_RUN_S", "30"))
    device = resolve_device(os.environ.get("GPULAB_DEVICE", "auto"))
    limiter = RateLimiter(int(os.environ.get("GPULAB_RATE_PER_MIN", "6")))
    cap_env = os.environ.get("GPULAB_DAILY_RUN_CAP")
    daily = DailyCap(int(cap_env) if cap_env else None)
    gpu_lock = threading.Lock()  # one measurement at a time: concurrent runs would contaminate
    prov = collect_provenance()

    app = FastAPI(
        title="GPU Inference Performance Lab",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
            max_age=600,
            # Chrome Private Network Access: lets a listed public HTTPS origin call a local API
            allow_private_network=True,
        )

    def client_id(request: Request) -> str:
        ip = request.client.host if request.client else "unknown"
        if trust_proxy and (fwd := request.headers.get("x-forwarded-for")):
            ip = fwd.split(",")[0].strip()
        return hashlib.sha256(ip.encode()).hexdigest()[:12]  # never log raw addresses

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        _json_log(
            event="request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round((time.perf_counter() - t0) * 1e3, 1),
            client=client_id(request),
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def support_matrix() -> dict[str, dict[str, str | None]]:
        return {
            n: {
                dt: impls.get(n).unsupported_reason(device, parse_dtype(dt))
                for dt in ("float32", "float16", "bfloat16")
            }
            for n in impls.names()
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "device": device.type,
            "device_name": device_name(device),
            "torch": torch.__version__,
            "git_commit": prov.git_commit,
            "busy": gpu_lock.locked(),
            "auth_required": api_key is not None,
            "limits": {
                "rows": [1, 64],
                "trials": [3, 20],
                "models": list(MODELS),
                "runs_per_minute": limiter.limit,
                "max_run_s": max_run_s,
                "daily_run_cap": daily.limit,
                "runs_remaining_today": daily.remaining(),
            },
        }

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "device": device.type,
            "candidates": CANDIDATES,
            "support": support_matrix(),
            "models": {k: v.to_dict() for k, v in MODELS.items()},
        }

    @app.post("/api/run")
    def run(
        req: RunRequest, request: Request, x_api_key: str | None = Header(None)
    ) -> dict[str, Any]:
        if api_key is not None and not hmac.compare_digest(x_api_key or "", api_key):
            raise HTTPException(401, "missing or invalid X-API-Key")
        ok, retry = limiter.allow(client_id(request))
        if not ok:
            raise HTTPException(
                429,
                f"rate limit: retry in {retry:.0f} s",
                headers={"Retry-After": str(int(retry) + 1)},
            )
        if req.impl not in CANDIDATES:
            raise HTTPException(400, f"impl must be one of {CANDIDATES}")
        dt = parse_dtype(req.dtype)
        reason = impls.get(req.impl).unsupported_reason(device, dt)
        if reason:
            raise HTTPException(400, f"{req.impl} is not available on this server: {reason}")
        if daily.remaining() == 0:
            raise HTTPException(
                429,
                "the demo's daily measurement budget is used up; it resets at 00:00 UTC",
                headers={"Retry-After": "3600"},
            )
        if not gpu_lock.acquire(blocking=False):
            raise HTTPException(
                429, "a measurement is already running; retry shortly", headers={"Retry-After": "5"}
            )
        if not daily.take():  # lost a race with another request for the last slot
            gpu_lock.release()
            raise HTTPException(
                429,
                "the demo's daily measurement budget is used up; it resets at 00:00 UTC",
                headers={"Retry-After": "3600"},
            )
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        try:
            cfg = BenchConfig(
                iters=req.trials * 10,
                trials=req.trials,
                burst=20,
                warmup_s=0.3,
                warmup_iters=5,
                device_iters=0,
                keep_samples=False,
            )
            recs = run_interleaved(
                ["eager", req.impl], MODELS[req.model], req.rows, device, dt, cfg, mode=req.mode
            )
        except Exception as exc:
            _json_log(event="run_failed", error=type(exc).__name__, detail=str(exc)[:200])
            raise HTTPException(500, f"measurement failed: {type(exc).__name__}") from exc
        finally:
            gpu_lock.release()
        elapsed = time.perf_counter() - t0
        base, cand = recs
        g, lo, hi = paired_ratio_ci(
            base.metrics["trial_medians_us"], cand.metrics["trial_medians_us"]
        )

        def side(r: Any) -> dict[str, Any]:
            h = r.metrics["host_us"]
            return {
                "impl": r.impl,
                "p50_us": h["p50"],
                "p95_us": h["p95"],
                "p99_us": h["p99"],
                "min_us": h["min"],
                "trials": len(r.metrics["trial_medians_us"]),
                "trial_medians_us": r.metrics["trial_medians_us"],
                "correctness": r.correctness,
            }

        _json_log(
            event="run",
            impl=req.impl,
            dtype=req.dtype,
            rows=req.rows,
            model=req.model,
            mode=req.mode,
            elapsed_s=round(elapsed, 2),
            speedup=round(g, 3),
        )
        return {
            "evidence": "live",
            "started_utc": started.isoformat(timespec="seconds"),
            "elapsed_s": round(elapsed, 3),
            "over_soft_cap": elapsed > max_run_s,
            "device": device.type,
            "device_name": device_name(device),
            "torch": torch.__version__,
            "git_commit": prov.git_commit,
            "request": req.model_dump(),
            "model": MODELS[req.model].to_dict(),
            "baseline": side(base),
            "candidate": side(cand),
            "speedup": {"geomean": g, "ci_low": lo, "ci_high": hi, "verdict": verdict(lo, hi)},
        }

    return app


logging.basicConfig(level=logging.INFO, format="%(message)s")
app = create_app()
