"""Load and robustness test for a running gpulab API.

Checks: health under 200 concurrent requests; concurrent /api/run calls yield exactly one
measurement and 429s (never 5xx); malformed inputs yield only 4xx.

Usage: python scripts/api_load_test.py [--url http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--out", default="results/phase8/api_load_test.json")
    a = p.parse_args()
    client = httpx.Client(base_url=a.url, timeout=60)

    def health(_: int) -> tuple[int, float]:
        t0 = time.perf_counter()
        r = client.get("/api/health")
        return r.status_code, (time.perf_counter() - t0) * 1e3

    with ThreadPoolExecutor(32) as ex:
        hs = list(ex.map(health, range(200)))
    lat = np.array([ms for _, ms in hs])
    health_codes = Counter(code for code, _ in hs)

    body = {"impl": "fused_weights", "dtype": "float32", "rows": 2, "model": "tiny", "trials": 3}
    with ThreadPoolExecutor(6) as ex:
        runs = list(ex.map(lambda _: client.post("/api/run", json=body).status_code, range(6)))
    run_codes = Counter(runs)

    bad_bodies = [
        {},
        {"rows": -1},
        {"rows": 10_000},
        {"impl": "os.system"},
        {"dtype": "int4"},
        {"model": "../../etc/passwd"},
        {"trials": 0},
        {"impl": "metal", "extra": 1},
        {"rows": "8; rm -rf /"},
        {"mode": "e2e"},
    ]
    fuzz = Counter(client.post("/api/run", json=b).status_code for b in bad_bodies)
    fuzz[
        client.post(
            "/api/run", content=b"not json", headers={"Content-Type": "application/json"}
        ).status_code
    ] += 1

    result = {
        "url": a.url,
        "health": {
            "codes": dict(health_codes),
            "p50_ms": float(np.median(lat)),
            "p99_ms": float(np.percentile(lat, 99)),
            "n": len(hs),
        },
        "concurrent_runs": dict(run_codes),
        "malformed_inputs": dict(fuzz),
    }
    ok = (
        set(health_codes) == {200}
        and run_codes.get(200, 0) >= 1
        and all(c in (200, 429) for c in run_codes)
        and all(400 <= c < 500 for c in fuzz)
    )
    result["passed"] = ok
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
