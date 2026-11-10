"use client";

import { useState } from "react";
import { ChartFrame, ForestPlot, LineChart, type Series } from "@/components/charts";
import { Card, ErrorState, EvidenceTag, LoadingPage, PageHeader, Section, Table } from "@/components/ui";
import { benchSeries, fmtUs, IMPL_COLOR, IMPL_LABEL, IMPL_ORDER, useSummary } from "@/lib/data";
import type { Summary } from "@/lib/types";

function Latency({ s }: { s: Summary }) {
  const [dtype, setDtype] = useState("float16");
  const by = benchSeries(s, { device: "mps", dtype, mode: "latency" });
  const mk = (q: "p50" | "p95" | "p99"): Series[] =>
    IMPL_ORDER.filter((k) => by.has(k)).map((k) => ({
      key: k,
      label: IMPL_LABEL[k] ?? k,
      color: IMPL_COLOR[k] ?? "var(--color-ink-2)",
      dashed: k === "eager",
      points: by.get(k)!.map((b) => ({ x: b.rows, y: b.host_us[q] })),
    }));
  const comps = s.comparisons
    .filter((c) => c.device === "mps" && c.dtype === dtype && c.mode === "latency" && c.d_model === 1024)
    .sort((a, b) => a.candidate.localeCompare(b.candidate) || a.rows - b.rows);
  const autocorr = s.benchmarks.filter((b) => b.mode === "latency" && b.device === "mps" && b.lag1_autocorr != null);
  const highAc = autocorr.filter((b) => (b.lag1_autocorr ?? 0) >= 0.2).length;
  const rep = s.baseline_phase2.repeatability;
  const ig = s.idle_gap;
  const series50 = mk("p50");

  return (
    <>
      <PageHeader eyebrow="Latency analysis" title="Synchronous request latency: medians, tails, and why the tails look this way">
        One request at a time, with the host synchronizing after each one: the latency a caller actually waits. Comparisons are interleaved
        trial by trial, so slow drift in the machine affects every implementation equally.
      </PageHeader>

      <div className="mb-4 flex items-center gap-3 text-sm">
        <span className="text-ink-2">Precision</span>
        <div className="flex rounded-md border border-line p-0.5" role="group" aria-label="Precision">
          {["float16", "bfloat16", "float32"].map((d) => (
            <button key={d} type="button" aria-pressed={dtype === d} onClick={() => setDtype(d)} className={`rounded px-2.5 py-1 text-xs transition-colors duration-150 ${dtype === d ? "bg-elevated text-ink" : "text-ink-2 hover:text-ink"}`}>
              {d}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {(["p50", "p95", "p99"] as const).map((q) => {
          const ser = mk(q);
          return (
            <ChartFrame
              key={q}
              title={`${q} latency vs rows`}
              subtitle={`Apple M4 GPU · ${dtype} · 1024×2816×4 · ${by.get("eager")?.[0]?.host_us.n ?? 0} requests per point`}
              legend={q === "p50" ? ser : undefined}
              table={<Table head={["rows", ...ser.map((x) => x.label)]} rows={(ser[0]?.points ?? []).map((p, i) => [p.x, ...ser.map((x) => fmtUs(x.points[i]?.y))])} />}
            >
              <LineChart series={ser} xScale="log2" xLabel="rows" yLabel="µs" height={220} ariaLabel={`${q} latency by rows`} />
            </ChartFrame>
          );
        })}
      </div>
      <p className="mt-2 text-xs text-ink-3">
        Each panel has its own y-axis. p99 from {by.get("eager")?.[0]?.host_us.n ?? 0} samples rests on only a handful of observations and is indicative.
      </p>

      <Section
        title="Paired speedup vs eager (synchronous latency)"
        description="Geometric mean of per-trial median ratios with a 95% bootstrap CI. The dashed line is 1× (no change). Synchronous latency is noisier than pipelined time, so more of these are not significant."
      >
        <ChartFrame
          title={`Speedup over eager · ${dtype} · latency mode`}
          subtitle="▲ CI entirely above 1 · ◆ CI spans 1 (n.s.) · ▼ CI entirely below 1"
          table={
            <Table
              head={["impl", "rows", "eager p50", "impl p50", "speedup", "95% CI", "verdict"]}
              rows={comps.map((c) => [c.candidate, c.rows, fmtUs(c.baseline_p50_us), fmtUs(c.candidate_p50_us), `${c.speedup_geomean.toFixed(2)}×`, `${c.ci_low.toFixed(2)}–${c.ci_high.toFixed(2)}`, c.verdict])}
            />
          }
        >
          <ForestPlot
            ariaLabel="Speedup with confidence intervals"
            rows={comps.map((c) => ({
              key: `${c.candidate}-${c.rows}`,
              label: `${IMPL_LABEL[c.candidate] ?? c.candidate} · ${c.rows} rows`,
              value: c.speedup_geomean,
              lo: c.ci_low,
              hi: c.ci_high,
              verdict: c.verdict,
              detail: `${fmtUs(c.baseline_p50_us)} → ${fmtUs(c.candidate_p50_us)}`,
            }))}
          />
        </ChartFrame>
      </Section>

      <Section title="Why the tails look this way" description="Measured properties of this machine, not of the kernels.">
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <div className="flex items-start justify-between">
              <div className="text-sm font-medium">Tails are clustered</div>
              <EvidenceTag kind="measured" />
            </div>
            <p className="mt-2 text-xs leading-relaxed text-ink-2">
              {highAc} of {autocorr.length} latency records have lag-1 autocorrelation ≥ 0.2: slow requests arrive in runs, not independently.
              Bootstrap CIs on those records are flagged unreliable, and comparisons rely on interleaved, paired trials instead.
            </p>
          </Card>
          <Card>
            <div className="flex items-start justify-between">
              <div className="text-sm font-medium">Separate runs disagree</div>
              <EvidenceTag kind="measured" />
            </div>
            <p className="mt-2 text-xs leading-relaxed text-ink-2">
              Five fresh processes, same configuration: p50 ranged {fmtUs(Math.min(...(rep?.p50_per_run_us ?? [0])))} –{" "}
              {fmtUs(Math.max(...(rep?.p50_per_run_us ?? [0])))} ({rep?.p50_max_over_min.toFixed(2)}× max/min, CV {rep?.p50_run_cv.toFixed(2)}). Numbers
              from separate runs are never compared directly.
            </p>
          </Card>
          <Card>
            <div className="flex items-start justify-between">
              <div className="text-sm font-medium">Idle time costs latency</div>
              <EvidenceTag kind="derived" />
            </div>
            <ul className="mt-2 space-y-1 text-xs text-ink-2">
              {ig &&
                Object.entries(ig.slowdown_vs_no_gap).map(([g, v]) => (
                  <li key={g} className="tnum">
                    {Number(g) / 1000} ms gap before each request: <span className="text-ink">{v.geomean.toFixed(2)}×</span> slower [{v.ci_low.toFixed(2)},{" "}
                    {v.ci_high.toFixed(2)}]
                  </li>
                ))}
            </ul>
            <p className="mt-2 text-xs text-ink-3">Paired over {ig?.trials} interleaved trials, eager, {ig?.rows} rows. The 2 ms effect replicated across two runs.</p>
          </Card>
        </div>
        {series50.length === 0 && <p className="text-sm text-ink-2">No latency-mode records for this precision.</p>}
      </Section>
    </>
  );
}

export default function Page() {
  const st = useSummary();
  if (st.status === "loading") return <LoadingPage />;
  if (st.status === "error") return <ErrorState title="Could not load measurements" message={st.error} onRetry={st.retry} />;
  return <Latency s={st.data} />;
}
