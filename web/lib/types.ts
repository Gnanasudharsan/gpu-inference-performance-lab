// Mirrors src/gpulab/summary.py. Every value is measured, derived or theoretical as labeled
// in Summary.evidence; nothing here is synthesized by the frontend.

export type Verdict = "faster" | "slower" | "no significant difference";

export interface HostStats {
  n: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  mean: number;
}

export interface Roofline {
  weight_bytes: number | null;
  arithmetic_intensity: number | null;
  achieved_gbps: number | null;
  nominal_gbps: number | null;
  floor_us: number | null;
  fraction_of_nominal: number | null;
}

export interface Bench {
  id: string;
  impl: string;
  device: string;
  device_name: string;
  dtype: string;
  rows: number;
  d_model: number;
  d_hidden: number;
  depth: number;
  mode: "latency" | "throughput" | "e2e";
  group: string | null;
  host_us: HostStats;
  p50_ci: [number, number] | null;
  ci_reliable: boolean | null;
  lag1_autocorr: number | null;
  device_p50_us: number | null;
  throughput_rows_per_s: number | null;
  trial_cv: number | null;
  trials: number;
  build_s: number | null;
  first_call_us: number | null;
  correctness: number | null;
  correct: boolean | null;
  roofline: Roofline | null;
  quantiles_host_us?: number[];
  git_commit: string | null;
  git_dirty: boolean | null;
  created_utc: string;
}

export interface Comparison {
  device: string;
  device_name: string;
  dtype: string;
  d_model: number;
  d_hidden: number;
  depth: number;
  rows: number;
  mode: string;
  baseline: string;
  candidate: string;
  baseline_p50_us: number;
  candidate_p50_us: number;
  speedup_geomean: number;
  ci_low: number;
  ci_high: number;
  trials: number;
  verdict: Verdict;
}

export interface Serving {
  id: string;
  impl: string;
  device: string;
  dtype: string;
  policy: string;
  arrival: string;
  rate_rps: number;
  deadline_us: number;
  requests: number;
  counts: Record<"ok" | "rejected" | "expired" | "error", number>;
  latency_us: { p50?: number; p95?: number; p99?: number; max?: number };
  queue_p50_us: number | null;
  queue_p99_us: number | null;
  service_p50_us: number | null;
  deadline_miss_rate: number;
  goodput_rows_per_s: number;
  throughput_rows_per_s: number;
  utilization: number;
  mean_batch_rows: number | null;
  padding_overhead: number | null;
  saturated: boolean;
  wake_lateness_p99_us: number | null;
  git_commit: string | null;
}

export interface Accelerator {
  backend: string;
  name: string;
  count: number;
  memory_bytes: number | null;
}

export interface Summary {
  schema: number;
  generated_utc: string;
  evidence: Record<string, string>;
  environment: {
    os: string;
    cpu: string;
    memory_gib: number;
    torch: string;
    python: string;
    accelerators: Accelerator[];
    git_commit: string | null;
  };
  benchmarks: Bench[];
  comparisons: Comparison[];
  regressions: Comparison[];
  serving: Serving[];
  envelope: {
    miss_budget: number;
    deadline_us: number;
    rates: number[];
    max_sustainable_rps: Record<string, number | null>;
  } | null;
  idle_gap: {
    rows: number;
    trials: number;
    median_of_trial_medians_us: Record<string, number>;
    slowdown_vs_no_gap: Record<string, { geomean: number; ci_low: number; ci_high: number; verdict: string }>;
    git_commit: string;
  } | null;
  kernels: {
    launches_per_forward: Record<string, { aten: number; custom: number | null; total: number; complete: boolean }>;
    cuda_profiles?: Record<string, { aten: number; cuda_kernels: number | null; cuda_kernel_us: number | null; gpu_forward_span_us?: number | null }>;
    metal_tuning: { results: Record<string, Record<string, number>>; device_name: string; git_commit: string } | null;
    metal_v1_vs_v2: Record<string, {
      p50_us: Record<string, number>;
      v2_over_v1: { geomean: number; min: number; max: number };
      v2_over_eager: { geomean: number; min: number; max: number };
      v1_over_eager: { geomean: number; min: number; max: number };
      launch_v1: { nsg: number; cols: number };
      launch_v2: { nsg: number; cols: number };
    }> | null;
  };
  baseline_phase2: { repeatability: { p50_per_run_us: number[]; p50_run_cv: number; p50_max_over_min: number } | null };
  correctness: { device: string; dtype: string; rows: number; impl: string; rel_l2: number; max_scaled: number; passed: boolean }[];
  unavailable: string[];
  devices?: string[];
  gpu_sessions?: {
    run: string;
    measurements_reported?: boolean;
    gpu: string | null;
    git_commit: string | null;
    steps: { step: string; status: "ok" | "failed" | "skipped"; reason?: string; seconds?: number }[];
  }[];
}
