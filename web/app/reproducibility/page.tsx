"use client";

import { Card, CodeBlock, ErrorState, EvidenceTag, LoadingPage, PageHeader, Section, Table } from "@/components/ui";
import { shortSha, useSummary } from "@/lib/data";
import type { Summary } from "@/lib/types";

const REPO = "https://github.com/Gnanasudharsan/gpu-inference-performance-lab";

function Repro({ s }: { s: Summary }) {
  const commits = [...new Set(s.benchmarks.map((b) => b.git_commit).concat(s.serving.map((x) => x.git_commit)))].filter(Boolean) as string[];
  const dirty = s.benchmarks.filter((b) => b.git_dirty).length;
  return (
    <>
      <PageHeader eyebrow="Reproducibility" title="Every number traces back to a commit, a machine and a command">
        Records carry hardware, software, Git commit and dirty-file state, power and thermal state, and the full benchmark configuration.
        Traces and inputs are seeded, so replays regenerate identically.
      </PageHeader>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <div className="mb-2 flex items-center justify-between">
            <div className="text-sm font-medium">Recorded environment</div>
            <EvidenceTag kind="measured" />
          </div>
          <Table
            head={["", ""]}
            rows={[
              ["Accelerators", s.environment.accelerators.map((a) => `${a.backend}: ${a.name}`).join("; ")],
              ["CPU / memory", `${s.environment.cpu} · ${s.environment.memory_gib} GiB`],
              ["OS", s.environment.os],
              ["Python / PyTorch", `${s.environment.python} / ${s.environment.torch}`],
              ["Record commits", commits.map(shortSha).join(", ")],
              ["Records from a dirty tree", String(dirty)],
              ["Dataset generated", new Date(s.generated_utc).toLocaleString()],
            ]}
          />
        </Card>
        <Card>
          <div className="text-sm font-medium">Reproduce from scratch</div>
          <div className="mt-3">
            <CodeBlock>{`git clone ${REPO}
cd gpu-inference-performance-lab
make install && make check          # lint, types, 227+ tests
.venv/bin/gpulab env                # provenance for your machine
.venv/bin/gpulab experiment experiments/apple_m4.toml   # ~25 min (M-series Mac)
.venv/bin/gpulab experiment experiments/nvidia.toml     # on a CUDA host
.venv/bin/gpulab analyze            # results/summary.json + ANALYSIS.md
cd web && npm ci && npm run build   # this dashboard, from your data`}</CodeBlock>
          </div>
        </Card>
      </div>
      <Section title="Experiment configurations" description="Declarative TOML. Work that cannot run on the current machine is recorded as skipped with a reason, never estimated.">
        <Table
          head={["config", "what it runs", "status"]}
          rows={[
            ["experiments/smoke.toml", "tiny CPU pipeline check (CI)", "runs in CI on every push"],
            ["experiments/apple_m4.toml", "full MPS + CPU matrix, serving sweep", "executed (results/phase3, phase4)"],
            ["experiments/repro_check.toml", "eager vs metal re-measurement", "executed: replicated within CIs"],
            ["experiments/nvidia_t4.toml", "T4 session: Triton, CUDA graphs, reduce-overhead, serving", "executed on Kaggle (results/gpu/kaggle-t4-*)"],
            ["experiments/nvidia.toml", "full NVIDIA matrix incl. bf16 (Ampere or newer)", "prepared, not executed"],
          ]}
        />
      </Section>
      <Section title="Methodology documents">
        <ul className="grid gap-2 text-sm sm:grid-cols-2">
          {[
            ["BENCHMARK_METHODOLOGY.md", "timing modes, bias controls, sample sizes"],
            ["CORRECTNESS.md", "reference, tolerances, checker validation"],
            ["PROFILING.md", "launch counting, Nsight plans"],
            ["SERVING.md", "replay model, policies, results"],
            ["ENGINEERING_REPORT.md", "narrative of wins, failures, limits"],
            ["KNOWN_LIMITATIONS.md", "what is not measured or not done"],
          ].map(([f, d]) => (
            <li key={f}>
              <a className="text-accent hover:underline" href={`${REPO}/blob/main/docs/${f}`}>
                {f}
              </a>{" "}
              <span className="text-ink-3">· {d}</span>
            </li>
          ))}
        </ul>
      </Section>
    </>
  );
}

export default function Page() {
  const st = useSummary();
  if (st.status === "loading") return <LoadingPage />;
  if (st.status === "error") return <ErrorState title="Could not load measurements" message={st.error} onRetry={st.retry} />;
  return <Repro s={st.data} />;
}
