"use client";

import { useEffect, useState } from "react";
import type { Bench, Comparison, Summary } from "./types";

type State = { status: "loading" } | { status: "error"; error: string } | { status: "ready"; data: Summary };

let cache: Promise<Summary> | null = null;

function load(): Promise<Summary> {
  if (!cache) {
    cache = fetch(`${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/data/summary.json`, { cache: "force-cache" })
      .then((r) => {
        if (!r.ok) throw new Error(`summary.json: HTTP ${r.status}`);
        return r.json() as Promise<Summary>;
      })
      .catch((e: unknown) => {
        cache = null; // allow retry
        throw e;
      });
  }
  return cache;
}

export function useSummary(): State & { retry: () => void } {
  const [state, setState] = useState<State>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let alive = true;
    setState({ status: "loading" });
    load()
      .then((data) => alive && setState({ status: "ready", data }))
      .catch((e: unknown) => alive && setState({ status: "error", error: e instanceof Error ? e.message : String(e) }));
    return () => {
      alive = false;
    };
  }, [attempt]);
  return { ...state, retry: () => setAttempt((a) => a + 1) };
}

// ---- selectors ----------------------------------------------------------------------------

export const IMPL_ORDER = ["eager", "fused_weights", "compile", "metal"] as const;

/** Series shown in line charts per device: at most four, in the validated palette order. */
export const DEVICE_SERIES: Record<string, readonly string[]> = {
  mps: IMPL_ORDER,
  cpu: ["eager", "fused_weights", "compile"],
  cuda: ["eager", "cuda_graph_triton", "compile_cudagraphs", "triton"],
};

export function seriesFor(device: string): readonly string[] {
  return DEVICE_SERIES[device] ?? IMPL_ORDER;
}

export const DEVICE_LABEL: Record<string, string> = { mps: "Apple M4 GPU", cuda: "NVIDIA GPU", cpu: "CPU" };

export const IMPL_LABEL: Record<string, string> = {
  eager: "eager (baseline)",
  fused_weights: "fused weights",
  compile: "torch.compile",
  metal: "Metal fused kernels",
  triton: "Triton fused kernels",
  cuda_graph_triton: "Triton + CUDA graph",
  cuda_graph_eager: "eager + CUDA graph",
  compile_cudagraphs: "compile (reduce-overhead)",
};

export const IMPL_COLOR: Record<string, string> = {
  eager: "var(--color-s-eager)",
  fused_weights: "var(--color-s-fused)",
  compile: "var(--color-s-compile)",
  metal: "var(--color-s-metal)",
  // colors encode role, so they stay consistent across devices: custom fused kernel = teal,
  // compiler = violet, CUDA-graph replay of the custom kernel = magenta (never charted with
  // fused_weights, which only appears on CPU/MPS views)
  triton: "var(--color-s-metal)",
  compile_cudagraphs: "var(--color-s-compile)",
  cuda_graph_triton: "var(--color-s-fused)",
};

export const DEFAULT_MODEL = { d_model: 1024, d_hidden: 2816, depth: 4 };

export function isModel(b: { d_model: number; d_hidden: number; depth: number }, m = DEFAULT_MODEL) {
  return b.d_model === m.d_model && b.d_hidden === m.d_hidden && b.depth === m.depth;
}

export function findComparison(
  s: Summary,
  q: Partial<Pick<Comparison, "device" | "dtype" | "mode" | "candidate" | "rows" | "d_model">>,
): Comparison | undefined {
  return s.comparisons.find((c) =>
    (Object.keys(q) as (keyof typeof q)[]).every((k) => c[k] === q[k]),
  );
}

export function benchSeries(
  s: Summary,
  q: { device: string; dtype: string; mode: string; d_model?: number },
): Map<string, Bench[]> {
  const out = new Map<string, Bench[]>();
  for (const b of s.benchmarks) {
    if (b.device !== q.device || b.dtype !== q.dtype || b.mode !== q.mode) continue;
    if (b.d_model !== (q.d_model ?? DEFAULT_MODEL.d_model)) continue;
    const arr = out.get(b.impl) ?? [];
    arr.push(b);
    out.set(b.impl, arr);
  }
  for (const arr of out.values()) arr.sort((a, b) => a.rows - b.rows);
  return out;
}

// ---- formatting ---------------------------------------------------------------------------

export function fmtUs(us: number | null | undefined): string {
  if (us == null || !Number.isFinite(us)) return "n/a";
  if (us >= 10_000) return `${(us / 1000).toFixed(1)} ms`;
  if (us >= 1000) return `${(us / 1000).toFixed(2)} ms`;
  return `${us.toFixed(0)} µs`;
}

export function fmtNum(v: number | null | undefined, digits = 0): string {
  if (v == null || !Number.isFinite(v)) return "n/a";
  return v.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function fmtX(v: number): string {
  return `${v.toFixed(2)}×`;
}

export function shortSha(sha: string | null | undefined): string {
  return sha ? sha.slice(0, 7) : "n/a";
}
