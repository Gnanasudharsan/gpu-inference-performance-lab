"use client";

import Link from "next/link";
import { ChartFrame, LineChart, type Series } from "@/components/charts";
import { Card, ErrorState, EvidenceTag, LoadingPage, PageHeader, Section, StatTile, Table } from "@/components/ui";
import { benchSeries, findComparison, fmtNum, fmtUs, IMPL_COLOR, IMPL_LABEL, IMPL_ORDER, shortSha, useSummary } from "@/lib/data";
import type { Summary } from "@/lib/types";

function Overview({ s }: { s: Summary }) {
  const q = { device: "mps", dtype: "float16", mode: "throughput", candidate: "metal", d_model: 1024 } as const;
  const metal = [1, 2, 4, 8].map((rows) => findComparison(s, { ...q, rows })).filter((c) => c != null);
  const best = metal.reduce((a, b) => (b.speedup_geomean > a.speedup_geomean ? b : a), metal[0]!);
  const worst = metal.reduce((a, b) => (b.speedup_geomean < a.speedup_geomean ? b : a), metal[0]!);
  const at8 = findComparison(s, { ...q, rows: 8 });
  const lp = s.kernels.launches_per_forward;
  const rf8 = s.benchmarks.find((b) => b.impl === "metal" && b.mode === "throughput" && b.dtype === "float16" && b.d_model === 1024 && b.rows === 8)?.roofline;
  const rfEager8 = s.benchmarks.find((b) => b.impl === "eager" && b.mode === "throughput" && b.dtype === "float16" && b.d_model === 1024 && b.rows === 8)?.roofline;
  const dyn = s.serving.filter((x) => x.policy === "dynamic" && x.arrival === "poisson");
  const direct = s.serving.filter((x) => x.policy === "direct" && x.arrival === "poisson");
  const peakDyn = dyn.reduce((a, b) => (b.goodput_rows_per_s > a.goodput_rows_per_s ? b : a), dyn[0]!);
  const peakDirect = direct.reduce((a, b) => (b.goodput_rows_per_s > a.goodput_rows_per_s ? b : a), direct[0]!);

  const bySeries = benchSeries(s, { device: "mps", dtype: "float16", mode: "throughput" });
  const series: Series[] = IMPL_ORDER.filter((k) => bySeries.has(k)).map((k) => ({
    key: k,
    label: IMPL_LABEL[k] ?? k,
    color: IMPL_COLOR[k] ?? "var(--color-ink-2)",
    dashed: k === "eager",
    points: bySeries.get(k)!.map((b) => ({ x: b.rows, y: b.host_us.p50 })),
  }));
  const gpu = s.environment.accelerators.find((a) => a.backend === "mps" || a.backend === "cuda");
  const cuda = ["triton", "cuda_graph_triton"].flatMap((cand) =>
    [1, 8].flatMap((rows) => {
      const c = findComparison(s, { device: "cuda", dtype: "float16", mode: "throughput", candidate: cand, rows, d_model: 1024 });
      return c ? [c] : [];
    }),
  );
  const cudaGpu = s.benchmarks.find((b) => b.device === "cuda")?.device_name ?? "NVIDIA";

  return (
    <>
      <PageHeader eyebrow="Overview" title="Small-batch GPU inference, measured end to end">
        A correct PyTorch baseline for a latency-sensitive RMSNorm → SwiGLU → residual network, pushed
        through algebraic fusion, <code className="font-mono">torch.compile</code>, and custom fused GPU kernels, then
        served under deadlines. Every number on this site comes from a record with hardware and Git provenance;
        comparisons are interleaved and paired with 95% confidence intervals.
      </PageHeader>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile
          label="Fused Metal kernels vs eager, 1–8 rows"
          value={`${worst.speedup_geomean.toFixed(2)}–${best.speedup_geomean.toFixed(2)}×`}
          evidence="derived"
          detail={
            <>
              Pipelined, fp16, 1024×2816×4. Best: {best.rows} rows [{best.ci_low.toFixed(2)}, {best.ci_high.toFixed(2)}]. Every
              CI lies above 1.
            </>
          }
        />
        <StatTile
          label="Kernel launches per forward"
          value={`${fmtNum(lp["eager/r8"]?.total)} → ${fmtNum(lp["metal/r8"]?.total)}`}
          evidence="measured"
          detail="Eager vs fused Metal path at 8 rows (2 fused kernels per block instead of 16 ops)."
        />
        <StatTile
          label="Weight-streaming bandwidth, 8 rows"
          value={fmtNum(rf8?.achieved_gbps, 0)}
          unit="GB/s"
          evidence="derived"
          detail={
            <>
              {fmtNum((rf8?.fraction_of_nominal ?? 0) * 100)}% of the {fmtNum(rf8?.nominal_gbps)} GB/s nominal peak{" "}
              <span className="text-warn">(theoretical)</span>; eager reaches {fmtNum(rfEager8?.achieved_gbps)} GB/s.
            </>
          }
        />
        <StatTile
          label="Peak serving goodput, dynamic batching"
          value={fmtNum(peakDyn?.goodput_rows_per_s)}
          unit="rows/s"
          evidence="measured"
          detail={
            <>
              {IMPL_LABEL[peakDyn?.impl ?? ""] ?? peakDyn?.impl} at {fmtNum(peakDyn?.rate_rps)} rps offered, 10 ms deadline. Direct execution peaks
              at {fmtNum(peakDirect?.goodput_rows_per_s)} rows/s.
            </>
          }
        />
      </div>

      <div className="mt-8 grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <ChartFrame
            title="Time per forward vs batch size"
            subtitle={
              <>
                Pipelined bursts (p50 of 20 interleaved trials) · fp16 · 1024×2816×4 · {gpu?.name} · commit{" "}
                <span className="font-mono">{shortSha(s.benchmarks[0]?.git_commit)}</span>
              </>
            }
            legend={series}
            table={
              <Table
                head={["rows", ...series.map((x) => x.label)]}
                rows={(series[0]?.points ?? []).map((p, i) => [p.x, ...series.map((x) => fmtUs(x.points[i]?.y))])}
              />
            }
          >
            <LineChart series={series} xScale="log2" xLabel="rows per request" yLabel="µs per forward" ariaLabel="Pipelined time per forward by batch size for each implementation" />
          </ChartFrame>
        </div>
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <div className="text-sm font-medium">Hardware & software</div>
            <EvidenceTag kind="measured" />
          </div>
          <dl className="space-y-2 text-xs">
            {[
              ["GPU", gpu?.name ?? "n/a"],
              ["CPU", `${s.environment.cpu}`],
              ["Memory", `${s.environment.memory_gib} GiB unified`],
              ["OS", s.environment.os],
              ["PyTorch", s.environment.torch],
              ["Python", s.environment.python],
              ["Records", `${s.benchmarks.length} benchmark · ${s.serving.length} serving`],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-4 border-b border-line/60 pb-2 last:border-0">
                <dt className="text-ink-3">{k}</dt>
                <dd className="text-right text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-4 rounded-lg border border-line bg-bg/60 p-3 text-xs leading-relaxed text-ink-2">
            <div className="mb-1 font-medium text-ink">Not measured here</div>
            <ul className="list-disc space-y-1 pl-4">
              {s.unavailable.map((u) => (
                <li key={u}>{u}</li>
              ))}
            </ul>
            {!(s.devices ?? []).includes("cuda") && (
              <p className="mt-2">
                Triton kernels and CUDA graphs are implemented and correctness-tested; their performance is pending NVIDIA hardware.
              </p>
            )}
          </div>
        </Card>
      </div>

      {cuda.length > 0 && (
        <Section title={`NVIDIA results · ${cudaGpu}`} description="Same workload, harness and paired method on an NVIDIA GPU. Pipelined fp16 time per forward, 1024×2816×4.">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {cuda.map((c) => (
              <StatTile
                key={`${c.candidate}-${c.rows}`}
                label={`${IMPL_LABEL[c.candidate] ?? c.candidate} vs eager · ${c.rows} rows`}
                value={`${c.speedup_geomean.toFixed(2)}×`}
                evidence="derived"
                detail={<>[{c.ci_low.toFixed(2)}, {c.ci_high.toFixed(2)}] · {c.verdict} · {fmtUs(c.baseline_p50_us)} → {fmtUs(c.candidate_p50_us)}</>}
              />
            ))}
          </div>
        </Section>
      )}

      <Section title="What the data says" description="The short version. Each claim links to the evidence.">
        <div className="grid gap-4 md:grid-cols-3">
          {[
            {
              t: "Fusion wins where launches and round trips dominate",
              d: `At 1–8 rows the fused kernels cut launches 64 → 8 and are ${worst.speedup_geomean.toFixed(2)}–${best.speedup_geomean.toFixed(2)}× faster. At 16 rows the advantage disappears, because the path falls back to vendor GEMM.`,
              href: "/kernels/",
            },
            {
              t: "Kernel speedups barely move serving",
              d: "End-to-end service is about 2 ms against about 0.8 ms of compute. The synchronous host↔GPU round trip, not the kernel, bounds serving on this machine.",
              href: "/serving/",
            },
            {
              t: "Tails are environmental and clustered",
              d: "About 7% of synchronous round trips stall for 3–10 ms (lag-1 autocorrelation up to 0.8). Paired, interleaved trials are what make the comparisons reproducible.",
              href: "/latency/",
            },
          ].map((c) => (
            <Link key={c.t} href={c.href} className="group rounded-xl border border-line bg-surface p-4 transition-colors duration-150 hover:border-accent/50">
              <div className="text-sm font-medium text-ink">{c.t}</div>
              <p className="mt-2 text-xs leading-relaxed text-ink-2">{c.d}</p>
              <div className="mt-3 text-xs text-accent">Evidence →</div>
            </Link>
          ))}
        </div>
        {at8 && (
          <p className="mt-4 text-xs text-ink-3">
            Reference comparison at 8 rows: eager {fmtUs(at8.baseline_p50_us)} vs fused Metal {fmtUs(at8.candidate_p50_us)} per forward,{" "}
            {at8.speedup_geomean.toFixed(2)}× [{at8.ci_low.toFixed(2)}, {at8.ci_high.toFixed(2)}] over {at8.trials} paired trials.
          </p>
        )}
      </Section>
    </>
  );
}

export default function Page() {
  const st = useSummary();
  if (st.status === "loading") return <LoadingPage />;
  if (st.status === "error") return <ErrorState title="Could not load measurements" message={st.error} onRetry={st.retry} />;
  return <Overview s={st.data} />;
}
