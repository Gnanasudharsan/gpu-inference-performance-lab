import importlib

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _client(monkeypatch, **env):
    monkeypatch.setenv("GPULAB_DEVICE", "cpu")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import gpulab.api as api

    importlib.reload(api)
    return TestClient(api.create_app()), api


TINY = {"impl": "fused_weights", "dtype": "float32", "rows": 2, "model": "tiny", "trials": 3}


def test_health_and_capabilities(monkeypatch):
    c, _ = _client(monkeypatch)
    h = c.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["device"] == "cpu"
    assert h["auth_required"] is False
    cap = c.get("/api/capabilities").json()
    assert "metal" in cap["support"]
    assert cap["support"]["metal"]["float16"]  # unsupported on cpu -> reason string


def test_run_returns_live_paired_measurement(monkeypatch):
    c, _ = _client(monkeypatch)
    r = c.post("/api/run", json=TINY)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["evidence"] == "live"
    assert d["device"] == "cpu"
    assert d["baseline"]["impl"] == "eager"
    assert d["candidate"]["correctness"]["passed"] is True
    assert d["speedup"]["ci_low"] <= d["speedup"]["geomean"] <= d["speedup"]["ci_high"]
    assert len(d["candidate"]["trial_medians_us"]) == 3
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "body",
    [
        {**TINY, "rows": 0},
        {**TINY, "rows": 65},
        {**TINY, "trials": 100},
        {**TINY, "model": "huge"},
        {**TINY, "dtype": "int8"},
        {**TINY, "code": "import os"},
    ],
)
def test_rejects_out_of_bounds_or_unknown_fields(monkeypatch, body):
    c, _ = _client(monkeypatch)
    assert c.post("/api/run", json=body).status_code == 422


def test_rejects_unsupported_and_unknown_impl(monkeypatch):
    c, _ = _client(monkeypatch)
    r = c.post("/api/run", json={**TINY, "impl": "metal"})
    assert r.status_code == 400
    assert "Apple GPUs" in r.json()["detail"]
    assert c.post("/api/run", json={**TINY, "impl": "eager"}).status_code == 400
    assert c.post("/api/run", json={**TINY, "impl": "rm -rf"}).status_code == 400


def test_api_key_required_when_configured(monkeypatch):
    c, _ = _client(monkeypatch, GPULAB_API_KEY="s3cret")
    assert c.get("/api/health").json()["auth_required"] is True
    assert c.post("/api/run", json=TINY).status_code == 401
    assert c.post("/api/run", json=TINY, headers={"X-API-Key": "nope"}).status_code == 401
    assert c.post("/api/run", json=TINY, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_rate_limit(monkeypatch):
    c, _ = _client(monkeypatch, GPULAB_RATE_PER_MIN="1")
    assert c.post("/api/run", json=TINY).status_code == 200
    r = c.post("/api/run", json=TINY)
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_rate_limiter_window():
    from gpulab.api import RateLimiter

    rl = RateLimiter(2, window_s=10)
    assert rl.allow("a", now=0)[0]
    assert rl.allow("a", now=1)[0]
    assert not rl.allow("a", now=2)[0]
    assert rl.allow("b", now=2)[0]
    assert rl.allow("a", now=10.5)[0]


def test_cors_allowlist(monkeypatch):
    c, _ = _client(monkeypatch, GPULAB_CORS_ORIGINS="https://example.test")
    ok = c.options(
        "/api/run",
        headers={"Origin": "https://example.test", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "https://example.test"
    bad = c.options(
        "/api/run", headers={"Origin": "https://evil.test", "Access-Control-Request-Method": "POST"}
    )
    assert bad.headers.get("access-control-allow-origin") is None
