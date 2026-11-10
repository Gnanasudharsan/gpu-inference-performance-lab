"use client";

import { ChartFrame, ForestPlot, HBarChart, Heatmap, LineChart, type Series } from "@/components/charts";
import { Card, DeviceToggle, ErrorState, EvidenceTag, LoadingPage, PageHeader, Section, Table, VerdictBadge } from "@/components/ui";
import { useState } from "react";
import { benchSeries, fmtNum, fmtUs, IMPL_COLOR, IMPL_LABEL, IMPL_ORDER, seriesFor, useSummary } from "@/lib/data";
import type { Summary } from "@/lib/types";

const SHORT: Record<string, string> = {
  eager: "eager",
  fused_weights: "fused",
  compile: "compile",
  metal: "metal",
  triton: "triton",
  cuda_graph_triton: "triton+graph",
  compile_cudagraphs: "compile+graphs",
};

function Kernels({ s }: { s: Summary }) {
  const [device, setDevice] = useState("mps");
  const isCuda = device === "cuda";
  const cp = s.kernels.cuda_profiles ?? {};
  const lp = s.kernels.launches_per_forward;
  const cudaBars = ["eager", "fused_weights", "compile", "triton", "cuda_graph_triton", "compile_cudagraphs"].flatMap((k) =>
    [8, 32].flatMap((r) => {
      const v = cp[`${k}/r${r}`];
      if (!v || v.cuda_kernels == null) return [];
      return [{ key: `${k}${r}`, label: `${SHORT[k] ?? k} · ${r} rows`, value: v.cuda_kernels, color: IMPL_COLOR[k] ?? "var(--color-ink-3)" }];
    }),
  );
  const mpsBars = IMPL_ORDER.flatMap((k) =>
    [8, 32].flatMap((r) => {
      const v = lp[`${k}/r${r}`];
      if (!v) return [];
      return [{ key: `${k}${r}`, label: `${SHORT[k] ?? k} · ${r} rows${v.complete ? "" : " *"}`, value: v.total, color: IMPL_COLOR[k] ?? "var(--color-ink-2)" }];
    }),
  );

  const bars = isCuda ? cudaBars : mpsBars;
  const by = benchSeries(s, { device, dtype: "float16", mode: "throughput" });
  const nominal = by.get("eager")?.[0]?.roofline?.nominal_gbps ?? null;
  const bw: Series[] = seriesFor(device).filter((k) => by.has(k)).map((k) => ({
    key: k,
    label: IMPL_LABEL[k] ?? k,
    color: IMPL_COLOR[k] ?? "var(--color-ink-2)",
    dashed: k === "eager",
    points: by.get(k)!.map((b) => ({ x: b.rows, y: b.roofline?.achieved_gbps ?? 0, label: `${fmtNum(b.roofline?.achieved_gbps, 1)} GB/s (${fmtNum((b.roofline?.fraction_of_nominal ?? 0) * 100)}%)` })),
  }));

  const custom = isCuda ? ["triton", "cuda_graph_triton"] : ["metal"];
  const metalComps = s.comparisons
    .filter((c) => custom.includes(c.candidate) && c.mode === "throughput" && c.device === device)
    .sort((a, b) => a.d_model - b.d_model || a.dtype.localeCompare(b.dtype) || a.rows - b.rows);

  const tuning = s.kernels.metal_tuning?.results ?? {};
  const tRows = Object.keys(tuning);
  const tCfgs = tRows.length ? Object.keys(tuning[tRows[0]!]!).sort() : [];
  const ab = s.kernels.metal_v1_vs_v2 ?? {};

  return (
    <>
      <PageHeader eyebrow="Kernel analysis" title="From 64 launches to 8: fused kernels for skinny products">
        At 1–8 rows every GEMM is memory-bound, so the floor is weight bytes ÷ bandwidth. The eager path spends its time in launches and
        intermediate round trips. The fused kernels fold RMSNorm into the products by linearity, (s·x⊙g)W = s·((x⊙g)W), and fuse SiLU·mul
        and the residual into epilogues: two kernels per block. They are Metal GEMV kernels on Apple GPUs and tiled Triton GEMMs on NVIDIA
        GPUs, optionally replayed as CUDA graphs.
      </PageHeader>

      <DeviceToggle devices={s.devices ?? ["mps"]} value={device} onChange={setDevice} />
      <div className="grid gap-4 lg:grid-cols-2">
        <ChartFrame title={isCuda ? "CUDA kernels per forward" : "Launches per forward"} subtitle={isCuda ? "Device kernels recorded by the PyTorch profiler (CUDA activity) · fp16 · 1024×2816×4" : "Measured with the profiler plus instrumented kernel launchers · fp16 · 1024×2816×4"} table={<Table head={["implementation", "launches"]} rows={bars.map((b) => [b.label, fmtNum(b.value)])} />}>
          <HBarChart bars={bars} format={(v) => fmtNum(v)} ariaLabel="Kernel launches per forward by implementation" />
          {!isCuda && <p className="mt-2 text-xs text-ink-3">* torch.compile: only ATen-visible launches are counted; Inductor-generated kernels are not visible to this method.</p>}
        </ChartFrame>
        <ChartFrame
          title="Achieved weight-streaming bandwidth"
          subtitle={
            <>
              Derived from weight bytes ÷ measured pipelined time · dashed reference is the vendor-nominal {fmtNum(nominal)} GB/s{" "}
              <span className="text-warn">(theoretical)</span>
            </>
          }
          legend={bw}
          table={<Table head={["rows", ...bw.map((x) => x.label)]} rows={(bw[0]?.points ?? []).map((p, i) => [p.x, ...bw.map((x) => x.points[i]?.label ?? "n/a")])} />}
        >
          <LineChart series={bw} xScale="log2" xLabel="rows" yLabel="GB/s" refLine={nominal ? { y: nominal, label: `nominal ${nominal} GB/s (theoretical)` } : undefined} ariaLabel="Achieved bandwidth by rows" />
        </ChartFrame>
      </div>

      <Section title={isCuda ? "Paired speedup of the fused Triton path (with and without CUDA graphs)" : "Paired speedup of the fused Metal path"} description="Pipelined time, interleaved trials, 95% bootstrap CI. It helps most where launches dominate (the small model, 2–8 rows), and not at all at 16 rows, where the path falls back to vendor GEMM.">
        <ChartFrame
          title={isCuda ? "Triton fused kernels vs eager" : "Metal fused kernels vs eager"}
          subtitle="▲ faster (CI above 1) · ◆ not significant · ▼ slower (CI below 1)"
          table={<Table head={["model", "dtype", "rows", "eager", "metal", "speedup", "95% CI", "verdict"]} rows={metalComps.map((c) => [`${c.d_model}×${c.d_hidden}`, c.dtype, c.rows, fmtUs(c.baseline_p50_us), fmtUs(c.candidate_p50_us), `${c.speedup_geomean.toFixed(2)}×`, `${c.ci_low.toFixed(2)}–${c.ci_high.toFixed(2)}`, c.verdict])} />}
        >
          <ForestPlot
            ariaLabel="Metal speedup with confidence intervals by configuration"
            rows={metalComps.map((c) => ({ key: `${c.candidate}-${c.d_model}-${c.dtype}-${c.rows}`, label: `${isCuda ? `${SHORT[c.candidate]} · ` : ""}${c.d_model} · ${c.dtype} · ${c.rows} rows`, value: c.speedup_geomean, lo: c.ci_low, hi: c.ci_high, verdict: c.verdict, detail: `${fmtUs(c.baseline_p50_us)} → ${fmtUs(c.candidate_p50_us)}` }))}
          />
        </ChartFrame>
      </Section>

      {!isCuda && (
      <div className="grid gap-4 lg:grid-cols-2">
        <ChartFrame title="Launch-configuration tuning (v2 kernels)" subtitle="Pipelined time relative to the best configuration for that row count (1.000 = best). Lighter is better." table={undefined}>
          <Heatmap
            rows={tCfgs}
            cols={tRows}
            rowLabel="simdgroups×columns"
            colLabel="rows"
            ariaLabel="Relative time by launch configuration and rows"
            value={(cfg, r) => {
              const col = tuning[r];
              if (!col || col[cfg] == null) return undefined;
              return col[cfg]! / Math.min(...Object.values(col));
            }}
            format={(v) => v.toFixed(3)}
          />
          <p className="mt-3 text-xs text-ink-3">Chosen: 8×2 for ≤ 4 rows, 8×1 for 5–8 rows. 4 columns at 8 rows is 27–30% slower (likely register pressure; not profiled).</p>
        </ChartFrame>
        <Card>
          <div className="mb-2 flex items-center justify-between">
            <div className="text-sm font-medium">Iteration: v1 → v2 (interleaved A/B, 30 trials)</div>
            <EvidenceTag kind="measured" />
          </div>
          <Table
            head={["rows", "eager", "v1", "v2", "v2 ÷ v1 (trial range)"]}
            rows={Object.entries(ab).map(([r, v]) => [r, fmtUs(v.p50_us.eager), fmtUs(v.p50_us.metal_v1), fmtUs(v.p50_us.metal_v2), `${v.v2_over_v1.geomean.toFixed(2)}× (${v.v2_over_v1.min.toFixed(2)}–${v.v2_over_v1.max.toFixed(2)})`])}
          />
          <p className="mt-3 text-xs leading-relaxed text-ink-2">
            v1 re-read every activation row for each output column and used runtime-bounded accumulator arrays, so it <span className="text-bad">regressed</span>{" "}
            to 0.77× eager at 8 rows. v2 specializes per row count (accumulators in registers) and reuses each activation load across the
            simdgroup&apos;s columns.
          </p>
        </Card>
      </div>
      )}

      <Section title="Regressions and limits" description="Reported with the same rigor as the wins.">
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <div className="mb-2 text-sm font-medium">Regressions (CI entirely below 1)</div>
            <ul className="space-y-1.5 text-xs text-ink-2">
              {s.regressions.map((c) => (
                <li key={`${c.device}${c.dtype}${c.mode}${c.candidate}${c.rows}`} className="flex flex-wrap items-center gap-2">
                  <VerdictBadge verdict="slower" />
                  <span>
                    {c.device}/{c.dtype} · {IMPL_LABEL[c.candidate] ?? c.candidate} · {c.rows} rows · {c.mode}:{" "}
                    <span className="tnum text-ink">{c.speedup_geomean.toFixed(2)}×</span> [{c.ci_low.toFixed(2)}, {c.ci_high.toFixed(2)}]
                  </span>
                </li>
              ))}
            </ul>
          </Card>
          <Card>
            <div className="mb-2 flex items-center justify-between">
              <div className="text-sm font-medium">NVIDIA paths</div>
              <EvidenceTag kind={(s.devices ?? []).includes("cuda") ? "measured" : "pending"} />
            </div>
            {(s.gpu_sessions ?? []).length > 0 ? (
              <ul className="space-y-2 text-xs text-ink-2">
                {(s.gpu_sessions ?? []).map((g) => (
                  <li key={g.run}>
                    <div className="text-ink">
                      {g.gpu ?? "unknown GPU"} · run <span className="font-mono">{g.run}</span> · commit <span className="font-mono">{(g.git_commit ?? "").slice(0, 7)}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1">
                      {g.steps.map((st) => (
                        <span key={st.step} title={st.reason ?? ""}>
                          <span aria-hidden="true" className={st.status === "ok" ? "text-good" : st.status === "failed" ? "text-bad" : "text-warn"}>
                            {st.status === "ok" ? "▲" : st.status === "failed" ? "▼" : "◆"}
                          </span>{" "}
                          {st.step}: {st.status}
                        </span>
                      ))}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs leading-relaxed text-ink-2">
                The Triton kernels fold RMSNorm into the K-loop of a tiled <code className="font-mono">tl.dot</code> GEMM, with two accumulators and nonlinear epilogues.
                They pass fp32/fp16 correctness under <code className="font-mono">TRITON_INTERPRET=1</code> in CI. CUDA graphs and{" "}
                <code className="font-mono">reduce-overhead</code> are implemented. None of these have been timed: no NVIDIA GPU session has been ingested yet.
              </p>
            )}
          </Card>
        </div>
      </Section>
    </>
  );
}

export default function Page() {
  const st = useSummary();
  if (st.status === "loading") return <LoadingPage />;
  if (st.status === "error") return <ErrorState title="Could not load measurements" message={st.error} onRetry={st.retry} />;
  return <Kernels s={st.data} />;
}
