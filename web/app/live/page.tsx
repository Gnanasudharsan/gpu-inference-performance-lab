"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ChartFrame, LineChart } from "@/components/charts";
import { Card, CodeBlock, EvidenceTag, PageHeader, Section, Table, VerdictBadge } from "@/components/ui";
import { fmtUs, IMPL_COLOR, IMPL_LABEL, shortSha } from "@/lib/data";
import type { Verdict } from "@/lib/types";

const DEFAULT_API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const STORE_KEY = "gpulab.apiUrl";

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

type Conn = { state: "checking" } | { state: "online"; health: Health } | { state: "offline"; error: string };

function readStoredUrl(): string {
  try {
    return window.localStorage.getItem(STORE_KEY) || DEFAULT_API;
  } catch {
    return DEFAULT_API;
  }
}

export default function LivePage() {
  const [apiUrl, setApiUrl] = useState(DEFAULT_API);
  const [draftUrl, setDraftUrl] = useState(DEFAULT_API);
  const [conn, setConn] = useState<Conn>({ state: "checking" });
  const [apiKey, setApiKey] = useState("");
  const [form, setForm] = useState({ impl: "metal", dtype: "float16", rows: 8, model: "default", mode: "throughput", trials: 10 });
  const [support, setSupport] = useState<Record<string, Record<string, string | null>> | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<RunResult[]>([]);

  useEffect(() => {
    const u = readStoredUrl();
    setApiUrl(u);
    setDraftUrl(u);
  }, []);

  const check = useCallback(async (url: string) => {
    setConn({ state: "checking" });
    setSupport(null);
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 4000);
      const r = await fetch(`${url}/api/health`, { signal: ctrl.signal });
      clearTimeout(t);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const health = (await r.json()) as Health;
      setConn({ state: "online", health });
      const c = await fetch(`${url}/api/capabilities`).then((x) => x.json());
      setSupport(c.support);
    } catch (e) {
      setConn({ state: "offline", error: e instanceof Error ? (e.name === "AbortError" ? "timed out" : e.message) : String(e) });
    }
  }, []);

  useEffect(() => {
    void check(apiUrl);
  }, [apiUrl, check]);

  const saveUrl = () => {
    const u = draftUrl.trim().replace(/\/+$/, "");
    try {
      window.localStorage.setItem(STORE_KEY, u);
    } catch {
      /* storage unavailable: keep in memory only */
    }
    setApiUrl(u);
  };

  const unsupported = support?.[form.impl]?.[form.dtype] ?? null;

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const r = await fetch(`${apiUrl}/api/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(apiKey ? { "X-API-Key": apiKey } : {}) },
        body: JSON.stringify(form),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${r.status}`);
      setResults((rs) => [body as RunResult, ...rs].slice(0, 5));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
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
          <div className="mb-3 text-sm font-medium">Backend</div>
          <label className="block text-xs text-ink-2" htmlFor="api-url">
            API URL
          </label>
          <div className="mt-1 flex gap-2">
            <input id="api-url" value={draftUrl} onChange={(e) => setDraftUrl(e.target.value)} className="min-w-0 flex-1 rounded-md border border-line bg-bg px-2.5 py-1.5 font-mono text-xs text-ink" spellCheck={false} />
            <button type="button" onClick={saveUrl} className="rounded-md border border-line px-3 text-xs text-ink hover:bg-elevated">
              Connect
            </button>
          </div>
          <div className="mt-3 text-xs" role="status" aria-live="polite">
            {conn.state === "checking" && <span className="text-ink-2">Checking {apiUrl} …</span>}
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
              <div className="text-ink-2">
                <span aria-hidden="true" className="text-bad">
                  ●
                </span>{" "}
                Offline ({conn.error}). No live execution is possible from this page right now.
              </div>
            )}
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
                {["metal", "fused_weights", "compile", "triton", "cuda_graph_triton"].map((i) => (
                  <option key={i} value={i}>
                    {IMPL_LABEL[i] ?? i}
                  </option>
                ))}
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
                disabled={conn.state !== "online" || running || !!unsupported}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-bg transition-opacity duration-150 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {running ? "Measuring on device…" : "Run measurement"}
              </button>
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

      {conn.state === "offline" && (
        <Section title="Run it locally" description="The API runs on any machine with gpulab installed, on the best local device (CUDA, Apple MPS, or CPU).">
          <CodeBlock>{`git clone https://github.com/Gnanasudharsan/gpu-inference-performance-lab
cd gpu-inference-performance-lab && make install
GPULAB_CORS_ORIGINS=http://localhost:3000 .venv/bin/gpulab serve --port 8000`}</CodeBlock>
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
