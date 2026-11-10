"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ChartFrame, LineChart } from "@/components/charts";
import { Card, CodeBlock, EvidenceTag, PageHeader, Section, Table, VerdictBadge } from "@/components/ui";
import { fmtUs, IMPL_COLOR, IMPL_LABEL, shortSha } from "@/lib/data";
import type { Verdict } from "@/lib/types";

/** Hosted backend baked in at build time (empty = this deployment has none). */
const HOSTED_API = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/+$/, "");
const LOCAL_API = "http://localhost:8000";
const STORE_KEY = "gpulab.apiUrl";

function isLocalUrl(u: string): boolean {
  try {
    const h = new URL(u).hostname;
    return h === "localhost" || h === "127.0.0.1" || h === "[::1]";
  } catch {
    return false;
  }
}

interface Health {
  status: string;
  device: string;
  device_name: string;
  torch: string;
  git_commit: string | null;
  busy: boolean;
  auth_required: boolean;
  limits: { rows: [number, number]; trials: [number, number]; models: string[]; runs_per_minute: number };
}

interface Side {
  impl: string;
  p50_us: number;
  p95_us: number;
  p99_us: number;
  min_us: number;
  trials: number;
  trial_medians_us: number[];
  correctness: { rel_l2: number; max_scaled: number; passed: boolean } | null;
}

interface RunResult {
  evidence: "live";
  started_utc: string;
  elapsed_s: number;
  device: string;
  device_name: string;
  torch: string;
  git_commit: string | null;
  request: { impl: string; dtype: string; rows: number; model: string; mode: string; trials: number };
  model: { d_model: number; d_hidden: number; depth: number };
  baseline: Side;
  candidate: Side;
  speedup: { geomean: number; ci_low: number; ci_high: number; verdict: Verdict };
}

type Conn =
  | { state: "idle" }
  | { state: "checking"; url: string; since: number }
  | { state: "online"; url: string; health: Health }
  | { state: "offline"; url: string; reason: string; fix: string | null };

function readStoredUrl(): string | null {
  try {
    return window.localStorage.getItem(STORE_KEY);
  } catch {
    return null;
  }
}

function storeUrl(u: string | null) {
  try {
    if (u) window.localStorage.setItem(STORE_KEY, u);
    else window.localStorage.removeItem(STORE_KEY);
  } catch {
    /* storage unavailable (private mode): keep the choice in memory only */
  }
}

/** Turn a failed health check into a human reason and, where possible, the exact fix. */
function diagnose(url: string, err: unknown, origin: string, timeoutS: number): { reason: string; fix: string | null } {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return { reason: "That is not a valid URL (expected something like https://example.com).", fix: null };
  }
  const local = isLocalUrl(url);
  if (origin.startsWith("https:") && parsed.protocol === "http:" && !local) {
    return { reason: "Browsers block http:// APIs from an https:// page (mixed content). Use the backend's https:// address.", fix: null };
  }
  if (err instanceof DOMException && err.name === "AbortError") {
    return local
      ? { reason: `No answer from ${parsed.host} within ${timeoutS} s. Is gpulab serve running?`, fix: `GPULAB_CORS_ORIGINS=${origin} gpulab serve --port ${parsed.port || "8000"}` }
      : { reason: `No answer within ${timeoutS} s. A hosted GPU backend that scaled to zero can take up to a minute to start; retrying usually works.`, fix: null };
  }
  if (err instanceof Error && err.message.startsWith("HTTP ")) {
    return { reason: `The server answered ${err.message}. It is reachable but not healthy.`, fix: null };
  }
  return local
    ? {
        reason: `Could not reach ${parsed.host}. Either no backend is running there, or it does not allow this page's origin (CORS). Safari blocks public sites from calling localhost; Chrome may ask for permission first.`,
        fix: `GPULAB_CORS_ORIGINS=${origin} gpulab serve --port ${parsed.port || "8000"}`,
      }
    : { reason: `Could not reach ${parsed.host}: the server may be down, or it does not allow this site's origin (CORS).`, fix: null };
}

export default function LivePage() {
  const [draftUrl, setDraftUrl] = useState(HOSTED_API || LOCAL_API);
  const [conn, setConn] = useState<Conn>({ state: "idle" });
  const [apiKey, setApiKey] = useState("");
  const [form, setForm] = useState({ impl: "metal", dtype: "float16", rows: 8, model: "default", mode: "throughput", trials: 10 });
  const [support, setSupport] = useState<Record<string, Record<string, string | null>> | null>(null);
  const [running, setRunning] = useState<number | null>(null); // start time while a run is in flight
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<RunResult[]>([]);
  const [origin, setOrigin] = useState("http://localhost:4310");
  const [now, setNow] = useState(() => Date.now());
  const apiUrl = conn.state === "idle" ? "" : conn.url;

  const check = useCallback(async (url: string) => {
    const local = isLocalUrl(url);
    const timeoutS = local ? 5 : 90;
    setConn({ state: "checking", url, since: Date.now() });
    setSupport(null);
    setError(null);
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutS * 1000);
    try {
      const r = await fetch(`${url}/api/health`, { signal: ctrl.signal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const health = (await r.json()) as Health;
      const c = await fetch(`${url}/api/capabilities`, { signal: ctrl.signal }).then((x) => x.json());
      setSupport(c.support);
      setConn({ state: "online", url, health });
      // default to the custom kernel that this backend's device supports
      setForm((f) => ({ ...f, impl: health.device === "cuda" ? "triton" : health.device === "mps" ? "metal" : "fused_weights" }));
    } catch (e) {
      setConn({ state: "offline", url, ...diagnose(url, e, window.location.origin, timeoutS) });
    } finally {
      clearTimeout(t);
    }
  }, []);

  useEffect(() => {
    setOrigin(window.location.origin);
    // Only probe something the visitor chose, or the deployment's hosted backend.
    // Never probe localhost uninvited: on a public page that fails for every visitor.
    const stored = readStoredUrl();
    const target = stored || HOSTED_API;
    if (target) {
      setDraftUrl(target);
      void check(target);
    }
  }, [check]);

  useEffect(() => {
    if (conn.state !== "checking" && running == null) return;
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, [conn.state, running]);

  const connect = (u: string) => {
    const url = u.trim().replace(/\/+$/, "");
    if (!url) return;
    setDraftUrl(url);
    storeUrl(url === HOSTED_API ? null : url);
    void check(url);
  };

  const disconnect = () => {
    storeUrl(null);
    setConn({ state: "idle" });
    setSupport(null);
  };

  const unsupported = support?.[form.impl]?.[form.dtype] ?? null;

  const run = async () => {
    if (conn.state !== "online") return;
    setRunning(Date.now());
    setError(null);
    try {
      const r = await fetch(`${apiUrl}/api/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(apiKey ? { "X-API-Key": apiKey } : {}) },
        body: JSON.stringify(form),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) {
        const detail = typeof body.detail === "string" ? body.detail : null;
        const retry = r.headers.get("Retry-After");
        if (r.status === 429) throw new Error(`${detail ?? "Busy or rate-limited."}${retry ? ` Try again in ${retry} s.` : ""}`);
        if (r.status === 401) throw new Error("This backend requires an API key. Enter it above.");
        if (r.status === 422) throw new Error("The backend rejected these settings (out of the allowed range).");
        throw new Error(detail ?? `The backend answered HTTP ${r.status}.`);
      }
      setResults((rs) => [body as RunResult, ...rs].slice(0, 5));
    } catch (e) {
      const msg = e instanceof TypeError ? "Lost the connection to the backend during the run. Check that it is still up and retry." : e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setRunning(null);
    }
  };

  const latest = results[0];
  const sel = "mt-1 w-full rounded-md border border-line bg-bg px-2.5 py-1.5 text-sm text-ink";

  return (
    <>
      <PageHeader eyebrow="Live execution" title="Run a bounded measurement on a real backend">
        Submits one paired comparison, eager vs. an optimized implementation, to a gpulab API server. That server measures it
        on its own device <em>now</em> and checks correctness against the float64 reference. Results here are labeled{" "}
        <EvidenceTag kind="live" />. Recorded measurements are on the other pages and are never shown as live.
      </PageHeader>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-sm font-medium">Backend</div>
            {conn.state !== "idle" && (
              <button type="button" onClick={disconnect} className="text-xs text-ink-3 hover:text-ink">
                Disconnect
              </button>
            )}
          </div>
          <div className="text-xs" role="status" aria-live="polite">
            {conn.state === "idle" && (
              <div className="space-y-2 text-ink-2">
                <div>
                  <span aria-hidden="true" className="text-ink-3">
                    ●
                  </span>{" "}
                  Not connected
                </div>
                <p className="leading-relaxed">
                  {HOSTED_API
                    ? "Connect to the hosted backend, or to one you run yourself."
                    : "This deployment has no hosted GPU backend. Connect to one you run yourself (one command, below); recorded measurements are on the other pages."}
                </p>
              </div>
            )}
            {conn.state === "checking" && (
              <div className="text-ink-2">
                <span aria-hidden="true" className="text-warn">
                  ●
                </span>{" "}
                Connecting to <span className="font-mono">{new URL(conn.url).host}</span> … {Math.max(0, Math.round((now - conn.since) / 1000))} s
                {!isLocalUrl(conn.url) && <div className="mt-1 text-ink-3">A hosted GPU backend may need up to a minute to wake from scale-to-zero.</div>}
              </div>
            )}
            {conn.state === "online" && (
              <div className="space-y-1 text-ink-2">
                <div>
                  <span aria-hidden="true" className="text-good">
                    ●
                  </span>{" "}
                  Online · <span className="text-ink">{conn.health.device_name}</span>
                </div>
                <div>
                  PyTorch {conn.health.torch} · commit <span className="font-mono">{shortSha(conn.health.git_commit)}</span>
                  {conn.health.busy ? " · busy" : ""}
                </div>
                <div>Limit: {conn.health.limits.runs_per_minute} runs/min per client</div>
              </div>
            )}
            {conn.state === "offline" && (
              <div className="space-y-2 text-ink-2">
                <div>
                  <span aria-hidden="true" className="text-bad">
                    ●
                  </span>{" "}
                  Can&apos;t reach <span className="font-mono">{conn.url}</span>
                </div>
                <p className="leading-relaxed">{conn.reason}</p>
                {conn.fix && (
                  <div>
                    <div className="mb-1 text-ink-3">Start it with:</div>
                    <code className="block break-all rounded bg-bg px-2 py-1.5 font-mono text-[11px] text-ink">{conn.fix}</code>
                  </div>
                )}
                <button type="button" onClick={() => void check(conn.url)} className="rounded-md border border-line px-3 py-1 text-xs text-ink hover:bg-elevated">
                  Retry
                </button>
              </div>
            )}
          </div>

          <div className="mt-4 space-y-2 border-t border-line pt-3">
            {HOSTED_API && conn.state === "idle" && (
              <button type="button" onClick={() => connect(HOSTED_API)} className="w-full rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-bg">
                Connect to hosted backend
              </button>
            )}
            {(conn.state === "idle" || (conn.state === "offline" && conn.url !== LOCAL_API)) && (
              <button type="button" onClick={() => connect(LOCAL_API)} className="w-full rounded-md border border-line px-3 py-1.5 text-xs text-ink hover:bg-elevated">
                Use a local backend ({LOCAL_API.replace("http://", "")})
              </button>
            )}
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                connect(draftUrl);
              }}
            >
              <label htmlFor="api-url" className="sr-only">
                Backend URL
              </label>
              <input id="api-url" value={draftUrl} onChange={(e) => setDraftUrl(e.target.value)} placeholder="https://your-backend.example" className="min-w-0 flex-1 rounded-md border border-line bg-bg px-2.5 py-1.5 font-mono text-xs text-ink" spellCheck={false} />
              <button type="submit" className="rounded-md border border-line px-3 text-xs text-ink hover:bg-elevated">
                Connect
              </button>
            </form>
          </div>
          {conn.state === "online" && conn.health.auth_required && (
            <div className="mt-3">
              <label className="block text-xs text-ink-2" htmlFor="api-key">
                API key (required by this server)
              </label>
              <input id="api-key" type="password" autoComplete="off" value={apiKey} onChange={(e) => setApiKey(e.target.value)} className={sel} />
            </div>
          )}
        </Card>

        <Card className="lg:col-span-2">
          <div className="mb-3 text-sm font-medium">Workload</div>
          <form
            className="grid grid-cols-2 gap-3 sm:grid-cols-3"
            onSubmit={(e) => {
              e.preventDefault();
              void run();
            }}
          >
            <label className="text-xs text-ink-2">
              Implementation
              <select className={sel} value={form.impl} onChange={(e) => setForm({ ...form, impl: e.target.value })}>
                {["metal", "triton", "cuda_graph_triton", "compile_cudagraphs", "fused_weights", "compile"].map((i) => {
                  const off = support ? support[i]?.[form.dtype] : null;
                  return (
                    <option key={i} value={i} disabled={!!off}>
                      {IMPL_LABEL[i] ?? i}
                      {off ? " (not on this backend)" : ""}
                    </option>
                  );
                })}
              </select>
            </label>
            <label className="text-xs text-ink-2">
              Precision
              <select className={sel} value={form.dtype} onChange={(e) => setForm({ ...form, dtype: e.target.value })}>
                {["float16", "bfloat16", "float32"].map((d) => (
                  <option key={d}>{d}</option>
                ))}
              </select>
            </label>
            <label className="text-xs text-ink-2">
              Model
              <select className={sel} value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })}>
                <option value="default">default · 1024×2816×4</option>
                <option value="small">small · 512×1408×4</option>
                <option value="tiny">tiny · 64×160×2</option>
              </select>
            </label>
            <label className="text-xs text-ink-2">
              Rows per request (1–64)
              <input type="number" min={1} max={64} className={sel} value={form.rows} onChange={(e) => setForm({ ...form, rows: Math.max(1, Math.min(64, Number(e.target.value) || 1)) })} />
            </label>
            <label className="text-xs text-ink-2">
              Mode
              <select className={sel} value={form.mode} onChange={(e) => setForm({ ...form, mode: e.target.value })}>
                <option value="throughput">pipelined (stable)</option>
                <option value="latency">synchronous latency</option>
              </select>
            </label>
            <label className="text-xs text-ink-2">
              Paired trials (3–20)
              <input type="number" min={3} max={20} className={sel} value={form.trials} onChange={(e) => setForm({ ...form, trials: Math.max(3, Math.min(20, Number(e.target.value) || 3)) })} />
            </label>
            <div className="col-span-2 flex flex-wrap items-center gap-3 sm:col-span-3">
              <button
                type="submit"
                disabled={conn.state !== "online" || running != null || !!unsupported}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-bg transition-opacity duration-150 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {running != null ? `Measuring on device… ${Math.max(0, Math.round((now - running) / 1000))} s` : "Run measurement"}
              </button>
              {conn.state !== "online" && <span className="text-xs text-ink-3">Connect to a backend to run a measurement.</span>}
              {unsupported && <span className="text-xs text-warn">Not available on this backend: {unsupported}</span>}
              {error && (
                <span role="alert" className="text-xs text-bad">
                  {error}
                </span>
              )}
            </div>
          </form>
        </Card>
      </div>

      {conn.state !== "online" && (
        <Section title="Run a backend yourself" description="The API runs on any machine with gpulab installed, on the best local device (CUDA, Apple MPS, or CPU).">
          <CodeBlock>{`git clone https://github.com/Gnanasudharsan/gpu-inference-performance-lab
cd gpu-inference-performance-lab && make install
GPULAB_CORS_ORIGINS=${origin} .venv/bin/gpulab serve --port 8000`}</CodeBlock>
          <p className="mt-3 text-sm text-ink-2">
            Recorded results from the development machine are in the{" "}
            <Link className="text-accent hover:underline" href="/benchmarks/">
              benchmark explorer
            </Link>{" "}
            (labeled measured, with the commit and date they were recorded).
          </p>
        </Section>
      )}

      {latest && (
        <Section title="Latest live result">
          <div className="grid gap-4 lg:grid-cols-3">
            <Card>
              <div className="mb-2 flex items-center justify-between">
                <div className="text-xs text-ink-2">
                  {IMPL_LABEL[latest.candidate.impl] ?? latest.candidate.impl} vs eager
                </div>
                <EvidenceTag kind="live" />
              </div>
              <div className="text-3xl font-semibold">{latest.speedup.geomean.toFixed(2)}×</div>
              <div className="mt-1 text-xs text-ink-2">
                95% CI [{latest.speedup.ci_low.toFixed(2)}, {latest.speedup.ci_high.toFixed(2)}] · {latest.candidate.trials} paired trials
              </div>
              <div className="mt-2">
                <VerdictBadge verdict={latest.speedup.verdict} />
              </div>
              <dl className="mt-4 space-y-1.5 text-xs">
                {[
                  ["Device", latest.device_name],
                  ["Measured at", new Date(latest.started_utc).toLocaleString()],
                  ["Wall time", `${latest.elapsed_s.toFixed(2)} s`],
                  ["Workload", `${latest.request.dtype} · ${latest.model.d_model}×${latest.model.d_hidden}×${latest.model.depth} · ${latest.request.rows} rows · ${latest.request.mode}`],
                  ["Correctness", latest.candidate.correctness?.passed ? `pass (rel. L2 ${latest.candidate.correctness.rel_l2.toExponential(1)})` : "FAILED"],
                  ["Server commit", shortSha(latest.git_commit)],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-3">
                    <dt className="text-ink-3">{k}</dt>
                    <dd className="text-right text-ink">{v}</dd>
                  </div>
                ))}
              </dl>
            </Card>
            <div className="lg:col-span-2">
              <ChartFrame
                title="Per-trial median time, interleaved"
                subtitle="Each trial measures both implementations back to back in rotated order; the speedup is the geometric mean of the per-trial ratios."
                legend={[
                  { key: "eager", label: "eager", color: IMPL_COLOR.eager!, dashed: true },
                  { key: "cand", label: latest.candidate.impl, color: IMPL_COLOR[latest.candidate.impl] ?? "var(--color-accent)" },
                ]}
                table={
                  <Table
                    head={["trial", "eager", latest.candidate.impl, "ratio"]}
                    rows={latest.baseline.trial_medians_us.map((b, i) => {
                      const c = latest.candidate.trial_medians_us[i] ?? NaN;
                      return [i + 1, fmtUs(b), fmtUs(c), `${(b / c).toFixed(2)}×`];
                    })}
                  />
                }
              >
                <LineChart
                  ariaLabel="Per-trial median time for eager and the candidate"
                  xLabel="trial"
                  yLabel="µs"
                  series={[
                    { key: "eager", label: "eager", color: IMPL_COLOR.eager!, dashed: true, points: latest.baseline.trial_medians_us.map((y, i) => ({ x: i + 1, y })) },
                    { key: "cand", label: latest.candidate.impl, color: IMPL_COLOR[latest.candidate.impl] ?? "var(--color-accent)", points: latest.candidate.trial_medians_us.map((y, i) => ({ x: i + 1, y })) },
                  ]}
                />
              </ChartFrame>
            </div>
          </div>
          {results.length > 1 && (
            <div className="mt-4">
              <Table
                caption="Previous live results in this browser session"
                head={["measured at", "impl", "workload", "eager p50", "impl p50", "speedup", "verdict"]}
                rows={results.map((r) => [
                  new Date(r.started_utc).toLocaleTimeString(),
                  r.candidate.impl,
                  `${r.request.dtype} · ${r.request.model} · ${r.request.rows} rows`,
                  fmtUs(r.baseline.p50_us),
                  fmtUs(r.candidate.p50_us),
                  `${r.speedup.geomean.toFixed(2)}× [${r.speedup.ci_low.toFixed(2)}, ${r.speedup.ci_high.toFixed(2)}]`,
                  <VerdictBadge key="v" verdict={r.speedup.verdict} />,
                ])}
              />
            </div>
          )}
        </Section>
      )}
    </>
  );
}
