"use client";

import { useState } from "react";
import { ChartFrame, LineChart, type Series } from "@/components/charts";
import { Card, ErrorState, EvidenceTag, LoadingPage, PageHeader, Section, Table } from "@/components/ui";
import { fmtNum, fmtUs, IMPL_COLOR, IMPL_LABEL, useSummary } from "@/lib/data";
import type { Serving, Summary } from "@/lib/types";

const POLICY_LABEL: Record<string, string> = { direct: "direct (no batching)", static8: "static batch of 8", dynamic: "dynamic batching" };
const POLICY_COLOR: Record<string, string> = { direct: "var(--color-s-eager)", static8: "var(--color-s-fused)", dynamic: "var(--color-s-metal)" };

function ServingPage({ s }: { s: Summary }) {
  const [impl, setImpl] = useState("metal");
  const impls = [...new Set(s.serving.map((x) => x.impl))];
  const sweep = s.serving.filter((x) => x.arrival === "poisson" && x.impl === impl);
  const policies = ["direct", "static8", "dynamic"].filter((p) => sweep.some((x) => x.policy === p));
  const mk = (f: (x: Serving) => number | null | undefined): Series[] =>
    policies.map((p) => ({
      key: p,
      label: POLICY_LABEL[p] ?? p,
      color: POLICY_COLOR[p] ?? "var(--color-ink-2)",
      dashed: p === "direct",
      points: sweep
        .filter((x) => x.policy === p)
        .sort((a, b) => a.rate_rps - b.rate_rps)
        .map((x) => ({ x: x.rate_rps, y: f(x) ?? 0 }))
        .filter((pt) => Number.isFinite(pt.y)),
    }));
  const goodput = mk((x) => x.goodput_rows_per_s);
  const miss = mk((x) => x.deadline_miss_rate * 100);
  const p99 = mk((x) => (x.counts.ok > 0 ? (x.latency_us.p99 ?? null) : null)).map((ser) => ({ ...ser, points: ser.points.filter((p) => p.y > 0) }));
  const bursty = s.serving.filter((x) => x.arrival === "bursty");
  const env = s.envelope;
  const deadline = sweep[0]?.deadline_us ?? 10_000;
  const tableFor = (ser: Series[], f: (v: number) => string) => (
    <Table head={["offered rps", ...ser.map((x) => x.label)]} rows={(ser.find((x) => x.points.length)?.points ?? []).map((p) => [p.x, ...ser.map((x) => f(x.points.find((q) => q.x === p.x)?.y ?? NaN))])} />
  );

  return (
    <>
      <PageHeader eyebrow="Serving" title="Latency-aware execution under open-loop load">
        Deterministic Poisson traces are replayed open-loop against a single-device server: one row per request, {deadline / 1000} ms
        deadline, bounded queue, expired requests dropped. Every request that does not finish on time counts as a miss, including ones
        that were shed or expired.
      </PageHeader>

      <div className="mb-4 flex flex-wrap items-center gap-3 text-sm">
        <span className="text-ink-2">Implementation</span>
        <div className="flex rounded-md border border-line p-0.5" role="group" aria-label="Implementation">
          {impls.map((i) => (
            <button key={i} type="button" aria-pressed={impl === i} onClick={() => setImpl(i)} className={`rounded px-2.5 py-1 text-xs transition-colors duration-150 ${impl === i ? "bg-elevated text-ink" : "text-ink-2 hover:text-ink"}`}>
              {IMPL_LABEL[i] ?? i}
            </button>
          ))}
        </div>
        <EvidenceTag kind="measured" />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartFrame title="Goodput vs offered load" subtitle="Rows completed within the deadline per second" legend={goodput} table={tableFor(goodput, (v) => fmtNum(v))}>
          <LineChart series={goodput} xLabel="offered load (requests/s)" yLabel="rows/s on time" ariaLabel="Goodput by offered rate per policy" />
        </ChartFrame>
        <ChartFrame title="Deadline miss rate vs offered load" subtitle="Share of all requests that missed (incl. shed and expired)" legend={miss} table={tableFor(miss, (v) => `${v.toFixed(1)}%`)}>
          <LineChart series={miss} xLabel="offered load (requests/s)" yLabel="% missed" yFormat={(v) => `${v.toFixed(0)}%`} yMax={100} ariaLabel="Deadline miss rate by offered rate per policy" />
        </ChartFrame>
        <ChartFrame title="p99 latency of completed requests" subtitle="Static batching completes nothing at low load (every request expires), so its points start at 300 rps" legend={p99} table={tableFor(p99, (v) => fmtUs(v))}>
          <LineChart series={p99} xLabel="offered load (requests/s)" yLabel="ms" yFormat={(v) => (v / 1000).toFixed(0)} refLine={{ y: deadline, label: `deadline ${deadline / 1000} ms` }} ariaLabel="p99 latency by offered rate" />
        </ChartFrame>
        <Card>
          <div className="text-sm font-medium">Reading these curves</div>
          <ul className="mt-2 space-y-2 text-xs leading-relaxed text-ink-2">
            <li>
              <span className="text-ink">Direct execution saturates first.</span> End-to-end service is about 2 ms (H2D, forward, synchronizing D2H),
              so beyond about 400–600 rps the queue grows and goodput collapses.
            </li>
            <li>
              <span className="text-ink">Dynamic batching tracks load.</span> It uses a batch of 1 when idle and larger batches as the queue builds, and goodput keeps rising.
            </li>
            <li>
              <span className="text-ink">Static batching fails at low load.</span> Filling 8 rows takes 8 ÷ rate (160 ms at 50 rps), so every request
              expires first.
            </li>
            <li>
              <span className="text-ink">Tails are bounded by the platform.</span> Queueing never exceeded the deadline (expired requests are dropped), but rare
              single-batch GPU stalls of 114–392 ms appear as p99 outliers. Each point is a single 3-second run.
            </li>
          </ul>
        </Card>
      </div>

      <Section title="Bursty arrivals" description="Same 300 rps mean, delivered in bursts at 4× the mean rate.">
        <Table
          head={["implementation", "policy", "miss rate", "p99 (completed)", "goodput rows/s", "mean batch"]}
          rows={bursty
            .sort((a, b) => a.impl.localeCompare(b.impl) || a.deadline_miss_rate - b.deadline_miss_rate)
            .map((x) => [IMPL_LABEL[x.impl] ?? x.impl, POLICY_LABEL[x.policy] ?? x.policy, `${(x.deadline_miss_rate * 100).toFixed(1)}%`, fmtUs(x.latency_us.p99), fmtNum(x.goodput_rows_per_s), fmtNum(x.mean_batch_rows, 1)])}
        />
      </Section>

      {env && (
        <Section
          title="Operating envelope"
          description={`Highest tested offered rate at which at most ${env.miss_budget * 100}% of requests missed the ${env.deadline_us / 1000} ms deadline. With 3–10 ms GPU round-trip stalls in about 7% of requests on this machine, the envelope is narrow.`}
        >
          <Table
            head={["implementation / policy", "max sustainable rate"]}
            rows={Object.entries(env.max_sustainable_rps)
              .sort()
              .map(([k, v]) => [k, v == null ? "none in tested range" : `${v} rps`])}
          />
        </Section>
      )}
    </>
  );
}

export default function Page() {
  const st = useSummary();
  if (st.status === "loading") return <LoadingPage />;
  if (st.status === "error") return <ErrorState title="Could not load measurements" message={st.error} onRetry={st.retry} />;
  return <ServingPage s={st.data} />;
}
