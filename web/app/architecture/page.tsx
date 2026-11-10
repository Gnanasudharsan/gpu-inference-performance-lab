import { Card, PageHeader, Table } from "@/components/ui";

function Box({ x, y, w, h, title, sub, accent }: { x: number; y: number; w: number; h: number; title: string; sub: string; accent?: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill="var(--color-elevated)" stroke={accent ? "var(--color-accent)" : "var(--color-line)"} />
      <text x={x + 12} y={y + 22} className="fill-ink text-[12px] font-medium">
        {title}
      </text>
      <text x={x + 12} y={y + 40} className="fill-ink-3 text-[10.5px]">
        {sub}
      </text>
    </g>
  );
}

function Arrow({ d, label, lx, ly }: { d: string; label?: string; lx?: number; ly?: number }) {
  return (
    <g>
      <path d={d} fill="none" stroke="var(--color-ink-3)" strokeWidth={1.25} markerEnd="url(#arrow)" />
      {label && (
        <text x={lx} y={ly} className="fill-ink-2 text-[10px]">
          {label}
        </text>
      )}
    </g>
  );
}

export default function Page() {
  return (
    <>
      <PageHeader eyebrow="Architecture" title="How the pieces fit">
        One Python package does the measuring. A static dashboard presents recorded results, and a separate, bounded API performs live runs.
        Recorded and live numbers never share a code path in the UI.
      </PageHeader>
      <Card>
        <div className="overflow-x-auto">
          <svg viewBox="0 0 900 430" className="min-w-[720px]" role="img" aria-label="System architecture diagram">
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="var(--color-ink-3)" />
              </marker>
            </defs>
            <text x={20} y={24} className="fill-ink-3 text-[10px] uppercase tracking-widest">
              offline: measure → record → analyze
            </text>
            <Box x={20} y={40} w={190} h={56} title="experiments/*.toml" sub="declarative matrix + sweeps" />
            <Box x={250} y={40} w={200} h={56} title="gpulab.experiment / bench" sub="interleaved, paired, provenance" accent />
            <Box x={490} y={40} w={170} h={56} title="results/*.jsonl" sub="records (measured)" />
            <Box x={700} y={40} w={180} h={56} title="gpulab analyze" sub="summary.json + ANALYSIS.md" />
            <Arrow d="M210,68 H246" />
            <Arrow d="M450,68 H486" />
            <Arrow d="M660,68 H696" />

            <text x={20} y={140} className="fill-ink-3 text-[10px] uppercase tracking-widest">
              core package (src/gpulab)
            </text>
            <Box x={20} y={156} w={160} h={56} title="workload" sub="weights, inputs, fp64 ref" />
            <Box x={200} y={156} w={160} h={56} title="impls" sub="eager · fused · compile" />
            <Box x={380} y={156} w={160} h={56} title="kernels/metal" sub="fused GEMV (MPS)" accent />
            <Box x={560} y={156} w={160} h={56} title="kernels/triton" sub="fused GEMM (CUDA)" />
            <Box x={740} y={156} w={140} h={56} title="serving" sub="replay, batching" />
            <Arrow d="M350,96 V152" />

            <text x={20} y={258} className="fill-ink-3 text-[10px] uppercase tracking-widest">
              presentation
            </text>
            <Box x={20} y={274} w={250} h={64} title="Static dashboard (this site)" sub="Next.js export · reads summary.json" accent />
            <Box x={330} y={274} w={250} h={64} title="gpulab API (FastAPI)" sub="auth · rate limit · 1 job at a time" />
            <Box x={640} y={274} w={240} h={64} title="Device" sub="Apple M4 GPU (MPS) · CUDA · CPU" />
            <Arrow d="M790,96 C790,240 150,220 150,270" label="summary.json (recorded)" lx={420} ly={232} />
            <Arrow d="M270,318 H326" label="POST /api/run" lx={262} ly={356} />
            <Arrow d="M580,306 H636" />
            <Arrow d="M636,326 H584" />
            <text x={588} y={356} className="fill-ink-2 text-[10px]">
              live result (labeled)
            </text>
            <text x={20} y={400} className="fill-ink-3 text-[10.5px]">
              CI: lint · mypy · CPU tests · Triton kernels via TRITON_INTERPRET=1 · experiment smoke run
            </text>
          </svg>
        </div>
      </Card>
      <div className="mt-6">
        <Table
          head={["module", "responsibility"]}
          rows={[
            ["gpulab.workload", "model config, deterministic weights and inputs, float64 reference, roofline quantities"],
            ["gpulab.impls", "execution modes behind one interface: eager, fused_weights, compile, metal, triton, cuda_graph_*"],
            ["gpulab.kernels.metal / triton_kernels", "fused kernels: norm folded into GEMV/GEMM, SiLU·mul and residual epilogues"],
            ["gpulab.correctness", "scale-normalized error metrics and calibrated per-dtype tolerances"],
            ["gpulab.bench / stats", "latency, e2e and pipelined modes; interleaving; percentiles, bootstrap CIs, autocorrelation"],
            ["gpulab.profiling", "launch counts per forward (ATen leaf dispatches plus instrumented custom launchers)"],
            ["gpulab.serving", "seeded traces, open-loop replay, direct/static/dynamic policies, bounded queue"],
            ["gpulab.experiment / analysis / summary", "TOML runner, paired comparisons with verdicts, evidence-labeled dataset"],
            ["gpulab.api", "bounded live-benchmark HTTP service"],
            ["gpulab.provenance / records", "hardware/software/Git/power provenance; JSONL and Parquet records"],
          ]}
        />
      </div>
    </>
  );
}
