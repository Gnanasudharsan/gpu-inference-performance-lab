import { Card, PageHeader } from "@/components/ui";

const DECISIONS = [
  {
    t: "Workload: RMSNorm → SwiGLU → residual, 1–64 rows",
    why: "At small batch every GEMM is memory-bound (~1–8 FLOP/byte), and eager issues 16 ops per block. Launch overhead and intermediate round trips are first-order costs, which is the regime real-time inference lives in. The block is the feed-forward core of most modern models.",
    rejected: "A full transformer, because attention and KV-cache dilute the small-batch story and mature libraries already cover them. A ReLU MLP, which has too little fusable structure.",
  },
  {
    t: "Measure on the GPU available; gate NVIDIA work explicitly",
    why: "The development machine is an Apple M4. Its GPU has real launch overhead, bandwidth limits and a custom-kernel path (Metal). CUDA paths are implemented and correctness-tested, but marked pending until an authorized NVIDIA run.",
    rejected: "Estimating CUDA numbers, which would be fabrication. Renting GPUs without an agreed budget.",
  },
  {
    t: "float64 CPU golden reference, scale-normalized tolerances",
    why: "A reference independent of every implementation under test, itself validated against NumPy. The tolerances are set 3–10× above the eager baseline's measured error, and the checker is shown to catch four injected bugs.",
    rejected: "Comparing against eager in the same dtype, which hides shared errors. Raw atol/rtol, which is sensitive to magnitude.",
  },
  {
    t: "Interleaved, paired comparisons with bootstrap CIs",
    why: "Separate runs on this laptop differed by up to 2.2× in p50. Rotating implementation order every trial and computing per-trial ratios makes conclusions reproducible: an independent rerun replicated the headline speedups within overlapping CIs.",
    rejected: "Comparing separately run numbers. Discarding outliers.",
  },
  {
    t: "Host latency and device time in separate passes",
    why: "Recording MPS events on every request was measured to perturb host latency. The user-facing number is measured uninstrumented.",
    rejected: "Subtracting device time from host time sample by sample.",
  },
  {
    t: "Custom Metal GEMV kernels only where they win (≤ 8 rows)",
    why: "They beat eager at 1–8 rows. At 16 rows the measured crossover sends products to vendor GEMM, keeping only fused elementwise kernels.",
    rejected: "A general Metal GEMM, which is a large effort to compete with Apple's GEMM, with little latency payoff.",
  },
  {
    t: "Triton for NVIDIA fusion; no hand-written CUDA or PTX",
    why: "Triton expresses the needed fusion at tile level: the norm folded into the K-loop, dual accumulators, nonlinear epilogues. No NVIDIA hardware was available to justify or validate lower-level code.",
    rejected: "CUDA C++ or PTX for appearance. Revisit if Nsight shows Triton below DRAM bandwidth at small M.",
  },
  {
    t: "Open-loop, single-threaded discrete-event replay; spin-wait idle",
    why: "Closed-loop load hides queueing, and a client thread fights the GIL. macOS sleep overshot by up to ~10 ms, so the replay spins (p99 wake lateness ~0.1 ms, no measured effect on GPU service) and reports its own fidelity.",
    rejected: "Threaded load generators. Trusting time.sleep.",
  },
  {
    t: "Dynamic batching with p90 service prediction and slack margin",
    why: "Waiting until exactly deadline − median service missed deadlines in tests because of jitter. Dynamic batching gave 4–5× the peak goodput of direct execution and the lowest miss rate under bursts.",
    rejected: "Static batching, which misses every deadline at low load. Unbounded queues.",
  },
  {
    t: "Static dashboard + separate execution API",
    why: "Recorded measurements are static files that any host can serve, with no backend required. Live execution is a bounded, authenticated, rate-limited API that runs only on a real device, and its results are always labeled live.",
    rejected: "A full-stack server that renders recorded data. Mixing recorded and live numbers in one view.",
  },
];

export default function Page() {
  return (
    <>
      <PageHeader eyebrow="Engineering decisions" title="What was chosen, why, and what was rejected">
        Condensed from <code className="font-mono">docs/DECISIONS.md</code>. Each decision is tied to a measurement or a constraint, not a
        preference.
      </PageHeader>
      <div className="grid gap-4 md:grid-cols-2">
        {DECISIONS.map((d, i) => (
          <Card key={d.t}>
            <div className="font-mono text-[11px] text-accent">D{String(i + 1).padStart(2, "0")}</div>
            <div className="mt-1 text-sm font-medium text-ink">{d.t}</div>
            <p className="mt-2 text-xs leading-relaxed text-ink-2">{d.why}</p>
            <p className="mt-2 text-xs leading-relaxed text-ink-3">
              <span className="text-ink-2">Rejected:</span> {d.rejected}
            </p>
          </Card>
        ))}
      </div>
    </>
  );
}
