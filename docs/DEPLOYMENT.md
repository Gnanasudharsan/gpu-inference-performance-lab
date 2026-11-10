# Deployment

## Status (2026-11-10)

| Component | Status | Where |
|---|---|---|
| Dashboard (static) | **Deployed and verified** | https://gpu-inference-performance-lab.vercel.app |
| GPU execution API | **Not deployed.** Manifests prepared; needs GPU-cloud credentials and an agreed spending cap | `deploy/Dockerfile`, `deploy/modal_app.py` |
| Local execution API | **Verified** on Apple M4 (MPS): `gpulab serve` | this repository |

The deployed dashboard shows only recorded measurements, labeled with their commit and hardware.

**Live page behavior.**
- *Probing:* it probes only the deployment's hosted backend (`NEXT_PUBLIC_API_URL`, baked in at build
  time) or a URL the visitor chose. It never probes `localhost` uninvited, because on a public page that
  fails for every visitor and can trigger a browser local-network permission prompt.
- *States:* not connected, then connecting (with an elapsed timer and a 90 s cold-start window for
  hosted backends), then online, or offline with a diagnosed reason and fix:
  - mixed content (an `http://` API from an `https://` page)
  - a timeout or cold start
  - an unreachable or CORS-blocked server, with the exact `gpulab serve` command for the page's origin
  - HTTP errors
- *Run failures are explained:* busy or rate-limited (with retry-after), a missing key, out-of-range
  settings, or a lost connection.
- *It never substitutes recorded numbers for live ones.*

## Architecture

```
 Browser ──HTTPS──► Vercel (static files: HTML/JS + summary.json)          no server code
    │
    └──HTTPS, X-API-Key──► GPU execution API (FastAPI in a container, 1 GPU)
                              ├─ bounded inputs (enumerated impl/dtype/model; rows ≤ 64; trials ≤ 20)
                              ├─ rate limit per client · one measurement at a time · timeout
                              └─ measures on its own device; returns evidence = "live"
```

The two halves are separated on purpose. The dashboard has no runtime dependencies and costs nothing
to host. The GPU service is the only component that can incur cost, so it carries every cost control.

## Hosting decisions

### Frontend: Vercel (chosen)

The site is a static export, so any static host works. Vercel was chosen because the account was
already authenticated, the Hobby tier is free, and it provides HTTPS, a CDN and custom security
headers via `vercel.json` (HSTS, `X-Frame-Options: DENY`, `nosniff`, referrer policy, permissions
policy). These were verified with `curl -I` on the production URL.

**Alternatives.** GitHub Pages is equivalent and would also work; it needs a base-path configuration.
S3 plus CloudFront would add IAM and billing setup for no benefit at this scale.

### GPU service: Modal L4 (recommended, not yet authorized)

Prices below come from third-party compilations dated 2026 and **must be re-checked on the providers'
own pricing pages before committing**. The sources disagreed in places.

| Option | Billing (reported) | Cold start (reported) | Fit |
|---|---|---|---|
| **Modal, L4** | ~$0.80/GPU-h, per second, no charge while idle; Starter plan includes $30/month credit | container start + image pull; scale-to-zero after `scaledown_window` | **Recommended.** Python-native deploy (`modal_app.py`), hard `max_containers=1`, secrets store, dashboard spend limit |
| Runpod Serverless (24 GB class) | ~$0.69/h flex, per second; **billing includes container start, model load and idle timeout** | vendor claims sub-200 ms for warm endpoints; independent reports give 8–30 s for infrequent ones and ~53 s for a first large cold start | Viable. Cold starts are billed, and the endpoint config lives outside the repo |
| Dedicated cloud VM (L4/A10G) | hourly, billed while running | none once up | Most stable latency, but billed 24/7 unless stopped. Use it only for the one-off benchmark run in [GPU_VALIDATION_PLAN.md](GPU_VALIDATION_PLAN.md) |

**Serverless GPUs do not give stable latency.**
- *Cold starts:* the first request after scale-to-zero pays for container start, CUDA context
  creation, weight generation and Triton autotuning. That is seconds, not microseconds.
- *Placement and noisy neighbors:* GPU type, host and neighbors can change between invocations.
- *The API and the benchmarks are separate things:* the API reports *its own* measurement, wall time
  and device each time, and the dashboard never compares a live serverless number with a recorded one.
- *Benchmark runs:* serious benchmarking belongs on a dedicated instance with recorded clocks
  ([GPU_VALIDATION_PLAN.md](GPU_VALIDATION_PLAN.md)), not on a serverless endpoint.

Sources consulted: [Modal pricing explained (Beam)](https://www.beam.cloud/blog/modal-pricing-explained),
[Modal GPU pricing 2026 (Spheron)](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/),
[Modal pricing calculator (usagepricing)](https://usagepricing.com/tools/pricing-calculator/modal),
[Runpod serverless pricing docs](https://docs.runpod.io/serverless/pricing),
[Runpod pricing](https://www.runpod.io/pricing-page),
[RunPod vs Modal vs Lambda (PremAI)](https://blog.premai.io/serverless-llm-deployment-runpod-vs-modal-vs-lambda-2026).

## Security controls (implemented in `gpulab.api`)

| Control | Implementation |
|---|---|
| No arbitrary code execution | Request schema is enumerated: impl and dtype from fixed lists, model ∈ {tiny, small, default}, bounded integers, `extra = "forbid"` |
| Authentication | Optional `GPULAB_API_KEY`, required for `/api/run` when set; constant-time comparison |
| Rate limiting | Sliding window per client (default 6 runs/min); `Retry-After` on 429 |
| Concurrency | One measurement at a time (a lock); concurrent runs would also contaminate timing. Busy → 429 |
| Spend bound | `GPULAB_DAILY_RUN_CAP`: a global cap on measurements per UTC day (30 in `modal_app.py`). Only runs that actually start count; once reached → 429 until 00:00 UTC. `/api/health` reports runs remaining |
| Bounded runtime | Inputs cap the work (≤ 20 trials × small bursts); `timeout=120` in Modal; reported `over_soft_cap` |
| CORS | Explicit allowlist (`GPULAB_CORS_ORIGINS`), including Chrome Private Network Access preflights (`allow_private_network`) for listed origins only |
| Logging | Structured JSON, one line per request. Client addresses are SHA-256 hashed and never logged raw |
| Secrets | Provider secret store (Modal Secret). Nothing in code, image or repository. A scan of the tracked tree and the build output found no secrets |
| Container | Non-root user, read-only app directory, healthcheck on `/api/health` |

## Cost caps (before any GPU deployment)

1. Workspace spending limit in the provider dashboard. Proposed: **USD 10/month**, to be agreed with
   the owner.
2. `max_containers=1`: at most one GPU billed at a time.
3. `scaledown_window=120`: scale to zero when idle.
4. `timeout=120` per request; rate limit per client; API key required in public deployments.

## Deployment checklist (GPU service)

```bash
# 0. owner authorizes: provider account + spending cap
pip install modal && modal token new
modal secret create gpulab-api GPULAB_API_KEY="$(openssl rand -hex 24)" \
    GPULAB_CORS_ORIGINS=https://gpu-inference-performance-lab.vercel.app
PYTHONPATH=src modal deploy deploy/modal_app.py            # prints the https://...modal.run URL
# 1. verify
curl -s https://<url>/api/health | jq .                      # device must be cuda, name = L4
curl -s -X POST https://<url>/api/run -H 'Content-Type: application/json' \
    -d '{"impl":"triton","rows":8}' -o /dev/null -w '%{http_code}\n'   # expect 401 without a key
curl -s -X POST https://<url>/api/run -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
    -d '{"impl":"triton","rows":8,"trials":5}' | jq '.speedup, .candidate.correctness.passed'
for i in $(seq 8); do curl -s -o /dev/null -w '%{http_code} ' ...; done   # expect 429s after the limit
# 2. point the dashboard at it (optional default)
cd web && NEXT_PUBLIC_API_URL=https://<url> npm run build && vercel deploy --prod
# 3. teardown when done
modal app stop gpulab-api
```

## Frontend deployment (as executed)

```bash
gpulab analyze && cd web && npm run build
# deploy the static export (out/ plus vercel.json with security headers)
vercel deploy --prod --yes
```

Verification performed on the production URL:
- `/`, `/live/` and `/kernels/` return 200; an unknown path returns 404
- `/data/summary.json` serves 154 benchmark records from commit `758dbd6`
- the security headers are present
- the Live page reports *offline* when no backend is reachable

A cross-origin run from the production origin to a local API was verified at the HTTP level with
`curl`: preflight 200 with allow-origin and allow-private-network, then a live Metal run at 1.28×
[1.18, 1.41]. The embedded test browser itself blocks public→localhost requests
(`ERR_BLOCKED_BY_CLIENT`), so the in-browser path was verified against the locally served dashboard.
