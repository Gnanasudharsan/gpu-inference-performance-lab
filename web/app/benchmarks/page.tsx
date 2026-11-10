"use client";

import { useMemo, useState } from "react";
import { ChartFrame, LineChart, type Series } from "@/components/charts";
import { Card, ErrorState, EvidenceTag, LoadingPage, PageHeader, Table } from "@/components/ui";
import { fmtNum, fmtUs, IMPL_COLOR, IMPL_LABEL, shortSha, useSummary } from "@/lib/data";
import type { Bench, Summary } from "@/lib/types";

type Key = "device" | "dtype" | "model" | "impl" | "mode" | "rows";
const KEYS: { key: Key; label: string }[] = [
  { key: "device", label: "Device" },
  { key: "dtype", label: "Precision" },
  { key: "model", label: "Model" },
  { key: "impl", label: "Implementation" },
  { key: "mode", label: "Mode" },
  { key: "rows", label: "Rows" },
];

const val = (b: Bench, k: Key): string =>
  k === "model" ? `${b.d_model}×${b.d_hidden}×${b.depth}` : k === "rows" ? String(b.rows) : String(b[k]);

type SortKey = "rows" | "p50" | "p99" | "cv";

function Explorer({ s }: { s: Summary }) {
  const [filters, setFilters] = useState<Record<Key, string>>({ device: "mps", dtype: "float16", model: "1024×2816×4", impl: "all", mode: "throughput", rows: "all" });
  const [sort, setSort] = useState<SortKey>("rows");
  const [selected, setSelected] = useState<string | null>(null);

  const options = useMemo(() => {
    const o = {} as Record<Key, string[]>;
    for (const { key } of KEYS) {
      const vs = [...new Set(s.benchmarks.map((b) => val(b, key)))];
      o[key] = key === "rows" ? vs.sort((a, b) => Number(a) - Number(b)) : vs.sort();
    }
    return o;
  }, [s]);

  const rows = useMemo(() => {
    const out = s.benchmarks.filter((b) => KEYS.every(({ key }) => filters[key] === "all" || val(b, key) === filters[key]));
    const by: Record<SortKey, (b: Bench) => number> = {
      rows: (b) => b.rows * 100 + ["eager", "fused_weights", "compile", "metal"].indexOf(b.impl),
      p50: (b) => b.host_us.p50,
      p99: (b) => b.host_us.p99,
      cv: (b) => b.trial_cv ?? 0,
    };
    return out.sort((a, b) => by[sort](a) - by[sort](b));
  }, [s, filters, sort]);

  const sel = rows.find((b) => b.id === selected) ?? rows.find((b) => b.quantiles_host_us);
  const sameConfig = sel
    ? s.benchmarks.filter((b) => b.group === sel.group && b.mode === sel.mode && b.quantiles_host_us)
    : [];
  const qSeries: Series[] = sameConfig.map((b) => ({
    key: b.impl,
    label: IMPL_LABEL[b.impl] ?? b.impl,
    color: IMPL_COLOR[b.impl] ?? "var(--color-ink-2)",
    dashed: b.impl === "eager",
    points: (b.quantiles_host_us ?? []).filter((_, i) => i >= 1 && i <= 99).map((y, i) => ({ x: i + 1, y })),
  }));

  const selCls = "mt-1 w-full rounded-md border border-line bg-bg px-2 py-1.5 text-sm text-ink";
  return (
    <>
      <PageHeader eyebrow="Benchmark explorer" title="Every recorded measurement, filterable">
        {s.benchmarks.length} benchmark records from interleaved runs. Each one is a record in{" "}
        <code className="font-mono">results/phase3/</code> with hardware, software and Git provenance. Select a row to see its full
        latency distribution against the other implementations measured in the same interleaved group.
      </PageHeader>

      <Card className="mb-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-7">
          {KEYS.map(({ key, label }) => (
            <label key={key} className="text-xs text-ink-2">
              {label}
              <select className={selCls} value={filters[key]} onChange={(e) => setFilters({ ...filters, [key]: e.target.value })}>
                <option value="all">all</option>
                {options[key].map((o) => (
                  <option key={o} value={o}>
                    {key === "impl" ? (IMPL_LABEL[o] ?? o) : o}
                  </option>
                ))}
              </select>
            </label>
          ))}
          <label className="text-xs text-ink-2">
            Sort by
            <select className={selCls} value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
              <option value="rows">rows, impl</option>
              <option value="p50">p50</option>
              <option value="p99">p99</option>
              <option value="cv">trial CV</option>
            </select>
          </label>
        </div>
      </Card>

      <div className="grid gap-4 xl:grid-cols-5">
        <Card className="xl:col-span-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="text-sm font-medium">{rows.length} records</div>
            <EvidenceTag kind="measured" />
          </div>
          {rows.length === 0 ? (
            <p className="py-8 text-center text-sm text-ink-2">No records match these filters.</p>
          ) : (
            <div className="max-h-[520px] overflow-auto">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-surface">
                  <tr className="border-b border-line text-ink-3">
                    {["impl", "rows", "dtype", "mode", "p50", "p95", "p99", "rows/s", "trial CV", "commit"].map((h) => (
                      <th key={h} scope="col" className="whitespace-nowrap px-2 py-2 font-normal">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="tnum">
                  {rows.map((b) => (
                    <tr
                      key={b.id}
                      tabIndex={0}
                      onClick={() => setSelected(b.id)}
                      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setSelected(b.id)}
                      aria-selected={sel?.id === b.id}
                      className={`cursor-pointer border-b border-line/60 transition-colors duration-150 ${sel?.id === b.id ? "bg-elevated" : "hover:bg-elevated/50"}`}
                    >
                      <td className="whitespace-nowrap px-2 py-1.5 text-ink">
                        <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: IMPL_COLOR[b.impl] ?? "var(--color-ink-3)" }} />
                        {b.impl}
                      </td>
                      <td className="px-2 py-1.5 text-ink-2">{b.rows}</td>
                      <td className="px-2 py-1.5 text-ink-2">{b.dtype}</td>
                      <td className="px-2 py-1.5 text-ink-2">{b.mode}</td>
                      <td className="px-2 py-1.5 font-mono text-ink">{fmtUs(b.host_us.p50)}</td>
                      <td className="px-2 py-1.5 font-mono text-ink-2">{fmtUs(b.host_us.p95)}</td>
                      <td className="px-2 py-1.5 font-mono text-ink-2">{fmtUs(b.host_us.p99)}</td>
                      <td className="px-2 py-1.5 font-mono text-ink-2">{fmtNum(b.throughput_rows_per_s)}</td>
                      <td className={`px-2 py-1.5 font-mono ${(b.trial_cv ?? 0) > 0.1 ? "text-warn" : "text-ink-2"}`}>{(b.trial_cv ?? 0).toFixed(3)}</td>
                      <td className="px-2 py-1.5 font-mono text-ink-3">{shortSha(b.git_commit)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="mt-3 text-xs text-ink-3">
            p-values are of per-request host latency (latency mode) or per-forward time within pipelined bursts (throughput mode). Trial CV
            above 0.1 is highlighted: those numbers drifted between trials.
          </p>
        </Card>

        <div className="xl:col-span-2">
          {sel && qSeries.length > 0 ? (
            <ChartFrame
              title="Latency distribution (quantile curve)"
              subtitle={
                <>
                  {sel.device_name} · {sel.dtype} · {sel.d_model}×{sel.d_hidden}×{sel.depth} · {sel.rows} rows · {sel.mode} · n=
                  {sel.host_us.n} per implementation
                </>
              }
              legend={qSeries}
              table={
                <Table
                  head={["percentile", ...qSeries.map((x) => x.label)]}
                  rows={[1, 10, 25, 50, 75, 90, 95, 99].map((pct) => [`p${pct}`, ...qSeries.map((x) => fmtUs(x.points[pct - 1]?.y))])}
                />
              }
            >
              <LineChart series={qSeries} xLabel="percentile" yLabel="µs" ariaLabel="Latency quantile curves for the selected configuration" yZero={false} xFormat={(v) => (v % 25 === 0 || v === 1 || v === 99 ? `p${v}` : "")} />
            </ChartFrame>
          ) : (
            <Card>
              <p className="text-sm text-ink-2">
                Select a record with stored samples to see its distribution. Throughput-mode records hold one sample per burst, so their
                quantile curves are coarse.
              </p>
            </Card>
          )}
          {sel && (
            <Card className="mt-4">
              <div className="mb-2 text-sm font-medium">Record {sel.id}</div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                {[
                  ["device", sel.device_name],
                  ["min / max", `${fmtUs(sel.host_us.min)} / ${fmtUs(sel.host_us.max)}`],
                  ["p50 95% CI", sel.p50_ci ? `${fmtUs(sel.p50_ci[0])} – ${fmtUs(sel.p50_ci[1])}${sel.ci_reliable ? "" : " (unreliable: autocorrelated)"}` : "n/a"],
                  ["lag-1 autocorr.", sel.lag1_autocorr?.toFixed(2) ?? "n/a"],
                  ["device-event p50", fmtUs(sel.device_p50_us)],
                  ["build time", sel.build_s != null ? `${sel.build_s.toFixed(2)} s` : "n/a"],
                  ["first call", fmtUs(sel.first_call_us)],
                  ["correctness rel. L2", sel.correctness?.toExponential(1) ?? "n/a"],
                  ["commit", `${shortSha(sel.git_commit)}${sel.git_dirty ? " (dirty)" : ""}`],
                  ["recorded", new Date(sel.created_utc).toLocaleString()],
                ].map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-ink-3">{k}</dt>
                    <dd className="text-right font-mono text-ink">{v}</dd>
                  </div>
                ))}
              </dl>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}

export default function Page() {
  const st = useSummary();
  if (st.status === "loading") return <LoadingPage />;
  if (st.status === "error") return <ErrorState title="Could not load measurements" message={st.error} onRetry={st.retry} />;
  return <Explorer s={st.data} />;
}
