"use client";

import { useCallback, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

// ---------------------------------------------------------------------------------------
// Shared frame: title, units/provenance, legend, and a chart <-> table toggle so no value is
// ever gated behind color or hover.
// ---------------------------------------------------------------------------------------

export interface LegendItem {
  key: string;
  label: string;
  color: string;
  dashed?: boolean;
}

export function Legend({ items }: { items: LegendItem[] }) {
  if (items.length < 2) return null;
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2" aria-label="Legend">
      {items.map((it) => (
        <li key={it.key} className="flex items-center gap-1.5">
          <svg width="18" height="8" aria-hidden="true">
            <line x1="1" y1="4" x2="17" y2="4" stroke={it.color} strokeWidth="2" strokeLinecap="round" strokeDasharray={it.dashed ? "4 3" : undefined} />
          </svg>
          {it.label}
        </li>
      ))}
    </ul>
  );
}

export function ChartFrame({
  title,
  subtitle,
  legend,
  table,
  children,
}: {
  title: string;
  subtitle?: ReactNode;
  legend?: LegendItem[];
  table?: ReactNode;
  children: ReactNode;
}) {
  const [view, setView] = useState<"chart" | "table">("chart");
  return (
    <figure className="rounded-xl border border-line bg-surface p-4 sm:p-5">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <figcaption className="min-w-0">
          <div className="text-sm font-medium text-ink">{title}</div>
          {subtitle && <div className="mt-0.5 text-xs text-ink-2">{subtitle}</div>}
        </figcaption>
        {table && (
          <div className="flex rounded-md border border-line p-0.5 text-xs" role="group" aria-label="View">
            {(["chart", "table"] as const).map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setView(v)}
                aria-pressed={view === v}
                className={`rounded px-2 py-0.5 capitalize transition-colors duration-150 ${view === v ? "bg-elevated text-ink" : "text-ink-2 hover:text-ink"}`}
              >
                {v}
              </button>
            ))}
          </div>
        )}
      </div>
      {legend && view === "chart" && (
        <div className="mb-2">
          <Legend items={legend} />
        </div>
      )}
      {view === "chart" ? children : <div className="max-h-96 overflow-auto">{table}</div>}
    </figure>
  );
}

export function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => e && setW(Math.floor(e.contentRect.width)));
    ro.observe(el);
    setW(Math.floor(el.getBoundingClientRect().width));
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

/** Clean ticks whose range *covers* [lo, hi] (rounded outward), so no mark exceeds the axis. */
function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const span = hi - lo;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => span / st <= count) ?? 10 * mag;
  const start = Math.floor(lo / step + 1e-9) * step;
  const end = Math.ceil(hi / step - 1e-9) * step;
  const out: number[] = [];
  for (let v = start; v <= end + step * 1e-6; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

function fmtTick(v: number): string {
  if (Math.abs(v) >= 10_000) return `${(v / 1000).toFixed(0)}K`;
  if (Math.abs(v) >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

// ---------------------------------------------------------------------------------------
// Line chart: multi-series, one y-axis, crosshair tooltip (mouse + keyboard), direct labels.
// ---------------------------------------------------------------------------------------

export interface Point {
  x: number;
  y: number;
  label?: string;
}

export interface Series extends LegendItem {
  points: Point[];
}

export function LineChart({
  series,
  xScale = "linear",
  xLabel,
  yLabel,
  yFormat = fmtTick,
  xFormat = fmtTick,
  height = 260,
  refLine,
  yZero = true,
  yMax,
  ariaLabel,
}: {
  series: Series[];
  xScale?: "linear" | "log2";
  xLabel: string;
  yLabel: string;
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  height?: number;
  refLine?: { y: number; label: string };
  yZero?: boolean;
  /** Fixed axis maximum (e.g. 100 for percentages) instead of rounding up past the data. */
  yMax?: number;
  ariaLabel: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const id = useId();
  const xs = useMemo(() => [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b), [series]);
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  if (refLine) ys.push(refLine.y);
  const yMaxRaw = Math.max(...ys, 0);
  const yMinRaw = yZero ? 0 : Math.min(...ys);
  const pad = (yMaxRaw - yMinRaw) * 0.04 || 1;
  const yTicks = yMax != null ? niceTicks(0, yMax, 5).filter((t) => t <= yMax) : niceTicks(yZero ? 0 : yMinRaw - pad, yMaxRaw + pad, 5);
  const yLo = yTicks[0] ?? 0;
  const yHi = yTicks[yTicks.length - 1] ?? 1;
  const directLabels = series.length <= 4;
  const m = { top: 12, right: directLabels && width > 480 ? 118 : 16, bottom: 40, left: 56 };
  const iw = Math.max(10, width - m.left - m.right);
  const ih = height - m.top - m.bottom;
  const tx = (x: number) => {
    const a = xs[0] ?? 0;
    const b = xs[xs.length - 1] ?? 1;
    if (a === b) return iw / 2;
    if (xScale === "log2") return ((Math.log2(x) - Math.log2(a)) / (Math.log2(b) - Math.log2(a))) * iw;
    return ((x - a) / (b - a)) * iw;
  };
  const ty = (y: number) => ih - ((y - yLo) / (yHi - yLo || 1)) * ih;

  const onMove = useCallback(
    (clientX: number, rect: DOMRect) => {
      const px = clientX - rect.left - m.left;
      let best = 0;
      let bd = Infinity;
      xs.forEach((x, i) => {
        const d = Math.abs(tx(x) - px);
        if (d < bd) {
          bd = d;
          best = i;
        }
      });
      setHover(best);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [xs, iw, m.left],
  );

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowRight") setHover((h) => Math.min(xs.length - 1, (h ?? -1) + 1));
    else if (e.key === "ArrowLeft") setHover((h) => Math.max(0, (h ?? xs.length) - 1));
    else if (e.key === "Escape") setHover(null);
  };

  // direct labels at each series end, nudged apart vertically
  const ends = series
    .map((s) => ({ s, p: s.points[s.points.length - 1] }))
    .filter((e): e is { s: Series; p: Point } => !!e.p)
    .map((e) => ({ ...e, y: ty(e.p.y) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    const prev = ends[i - 1]!;
    const cur = ends[i]!;
    if (cur.y - prev.y < 13) cur.y = prev.y + 13;
  }

  // thin x tick labels so they never collide (always keep the last one)
  const xTickLabels: number[] = [];
  const minGap = 30;
  xs.forEach((x, i) => {
    const px = tx(x);
    const prev = xTickLabels[xTickLabels.length - 1];
    const isLast = i === xs.length - 1;
    if (prev == null || px - tx(prev) >= minGap) xTickLabels.push(x);
    else if (isLast) {
      xTickLabels.pop();
      xTickLabels.push(x);
    }
  });

  const hx = hover != null ? xs[hover] : undefined;
  return (
    <div ref={ref} className="relative w-full">
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={ariaLabel}
          tabIndex={0}
          onKeyDown={onKey}
          onMouseMove={(e) => onMove(e.clientX, e.currentTarget.getBoundingClientRect())}
          onMouseLeave={() => setHover(null)}
          onBlur={() => setHover(null)}
          className="block outline-none"
        >
          <g transform={`translate(${m.left},${m.top})`}>
            {yTicks.map((t) => (
              <g key={t}>
                <line x1={0} x2={iw} y1={ty(t)} y2={ty(t)} stroke="var(--color-grid)" strokeWidth={1} />
                <text x={-8} y={ty(t)} dy="0.32em" textAnchor="end" className="tnum fill-ink-3 text-[11px]">
                  {yFormat(t)}
                </text>
              </g>
            ))}
            <line x1={0} x2={iw} y1={ih} y2={ih} stroke="var(--color-line)" />
            {xTickLabels.map((x) => (
              <text key={x} x={tx(x)} y={ih + 16} textAnchor="middle" className="tnum fill-ink-3 text-[11px]">
                {xFormat(x)}
              </text>
            ))}
            <text x={iw / 2} y={ih + 34} textAnchor="middle" className="fill-ink-2 text-[11px]">
              {xLabel}
            </text>
            <text transform={`translate(${-44},${ih / 2}) rotate(-90)`} textAnchor="middle" className="fill-ink-2 text-[11px]">
              {yLabel}
            </text>
            {refLine && (
              <g>
                <line x1={0} x2={iw} y1={ty(refLine.y)} y2={ty(refLine.y)} stroke="var(--color-ink-3)" strokeDasharray="2 4" />
                <text x={4} y={ty(refLine.y) - 5} className="fill-ink-3 text-[10px]">
                  {refLine.label}
                </text>
              </g>
            )}
            {series.map((s) => (
              <path
                key={s.key}
                d={s.points.map((p, i) => `${i ? "L" : "M"}${tx(p.x)},${ty(p.y)}`).join("")}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
                strokeDasharray={s.dashed ? "6 4" : undefined}
              />
            ))}
            {series.map((s) =>
              s.points.map((p) => (
                <circle key={`${s.key}-${p.x}`} cx={tx(p.x)} cy={ty(p.y)} r={hx === p.x ? 5 : 3.5} fill={s.color} stroke="var(--color-surface)" strokeWidth={2} />
              )),
            )}
            {directLabels && width > 480 &&
              ends.map(({ s, p, y }) => (
                <text key={s.key} x={tx(p.x) + 10} y={y} dy="0.32em" className="fill-ink-2 text-[11px]">
                  {s.label}
                </text>
              ))}
            {hx != null && <line x1={tx(hx)} x2={tx(hx)} y1={0} y2={ih} stroke="var(--color-ink-3)" strokeWidth={1} />}
          </g>
        </svg>
      )}
      {hx != null && width > 0 && (
        <div
          role="status"
          aria-live="polite"
          id={id}
          className="pointer-events-none absolute top-2 z-10 min-w-40 rounded-lg border border-line bg-elevated px-3 py-2 text-xs shadow-xl"
          style={{ left: Math.min(width - 190, Math.max(0, m.left + tx(hx) + 12)) }}
        >
          <div className="mb-1 text-ink-2">
            {xLabel}: <span className="tnum text-ink">{xFormat(hx)}</span>
          </div>
          {series
            .map((s) => ({ s, p: s.points.find((p) => p.x === hx) }))
            .filter((e) => e.p)
            .sort((a, b) => (b.p!.y ?? 0) - (a.p!.y ?? 0))
            .map(({ s, p }) => (
              <div key={s.key} className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5 text-ink-2">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
                  {s.label}
                </span>
                <span className="tnum font-mono text-ink">{p!.label ?? yFormat(p!.y)}</span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------
// Forest plot: speedup with 95% CI, reference line at 1x, verdict icon + label per row.
// ---------------------------------------------------------------------------------------

export interface ForestRow {
  key: string;
  label: string;
  value: number;
  lo: number;
  hi: number;
  verdict: "faster" | "slower" | "no significant difference";
  detail?: string;
}

export const VERDICT_STYLE = {
  faster: { color: "var(--color-good)", icon: "▲", text: "faster" },
  slower: { color: "var(--color-bad)", icon: "▼", text: "slower" },
  "no significant difference": { color: "var(--color-warn)", icon: "◆", text: "n.s." },
} as const;

export function ForestPlot({ rows, ariaLabel }: { rows: ForestRow[]; ariaLabel: string }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);
  const rowH = 26;
  const labelW = Math.min(170, Math.max(110, width * 0.3));
  const m = { top: 8, right: 64, bottom: 30, left: labelW };
  const iw = Math.max(10, width - m.left - m.right);
  const lo = Math.min(0.8, ...rows.map((r) => r.lo));
  const hi = Math.max(1.3, ...rows.map((r) => r.hi));
  const ticks = niceTicks(lo, hi, 5);
  const a = ticks[0] ?? lo;
  const b = ticks[ticks.length - 1] ?? hi;
  const tx = (v: number) => ((v - a) / (b - a)) * iw;
  const height = m.top + m.bottom + rows.length * rowH;
  return (
    <div ref={ref} className="w-full">
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={ariaLabel}>
          <g transform={`translate(${m.left},${m.top})`}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={tx(t)} x2={tx(t)} y1={0} y2={rows.length * rowH} stroke="var(--color-grid)" />
                <text x={tx(t)} y={rows.length * rowH + 16} textAnchor="middle" className="tnum fill-ink-3 text-[11px]">
                  {t.toFixed(2)}×
                </text>
              </g>
            ))}
            <line x1={tx(1)} x2={tx(1)} y1={-4} y2={rows.length * rowH} stroke="var(--color-ink-3)" strokeDasharray="3 3" />
            {rows.map((r, i) => {
              const v = VERDICT_STYLE[r.verdict];
              const y = i * rowH + rowH / 2;
              return (
                <g
                  key={r.key}
                  onMouseEnter={() => setHover(r.key)}
                  onMouseLeave={() => setHover(null)}
                  tabIndex={0}
                  onFocus={() => setHover(r.key)}
                  onBlur={() => setHover(null)}
                  aria-label={`${r.label}: ${r.value.toFixed(2)}x, 95% CI ${r.lo.toFixed(2)} to ${r.hi.toFixed(2)}, ${v.text}`}
                >
                  <rect x={-labelW} y={i * rowH} width={width} height={rowH} fill={hover === r.key ? "var(--color-elevated)" : "transparent"} />
                  <text x={-10} y={y} dy="0.32em" textAnchor="end" className="fill-ink-2 text-[11px]">
                    {r.label}
                  </text>
                  <line x1={tx(r.lo)} x2={tx(r.hi)} y1={y} y2={y} stroke={v.color} strokeWidth={2} strokeLinecap="round" />
                  <circle cx={tx(r.value)} cy={y} r={4.5} fill={v.color} stroke="var(--color-surface)" strokeWidth={2} />
                  <text x={iw + 8} y={y} dy="0.32em" className="tnum fill-ink text-[11px]">
                    <tspan fill={v.color}>{v.icon}</tspan> {r.value.toFixed(2)}×
                  </text>
                  {hover === r.key && r.detail && (
                    <text x={tx(r.hi) + 8 > iw - 140 ? tx(r.lo) - 8 : tx(r.hi) + 8} y={y - 8} textAnchor={tx(r.hi) + 8 > iw - 140 ? "end" : "start"} className="tnum fill-ink-2 text-[10px]">
                      {r.detail}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        </svg>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------
// Heatmap (sequential, one hue): relative time vs best, lower = better.
// ---------------------------------------------------------------------------------------

export function Heatmap({
  rows,
  cols,
  value,
  format,
  ariaLabel,
  rowLabel,
  colLabel,
}: {
  rows: string[];
  cols: string[];
  value: (r: string, c: string) => number | undefined;
  format: (v: number) => string;
  ariaLabel: string;
  rowLabel: string;
  colLabel: string;
}) {
  const vals = rows.flatMap((r) => cols.map((c) => value(r, c))).filter((v): v is number => v != null);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  // single blue ramp, light (best) -> dark (worst) on the dark surface
  const ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#104281"];
  const color = (v: number) => ramp[Math.min(ramp.length - 1, Math.floor(((v - lo) / (hi - lo || 1)) * (ramp.length - 1e-9)))]!;
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-0.5 text-xs" aria-label={ariaLabel}>
        <thead>
          <tr>
            <th className="px-2 py-1 text-left font-normal text-ink-3">{rowLabel} \ {colLabel}</th>
            {cols.map((c) => (
              <th key={c} className="px-2 py-1 text-center font-normal text-ink-2">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r}>
              <th scope="row" className="whitespace-nowrap px-2 py-1 text-left font-mono font-normal text-ink-2">
                {r}
              </th>
              {cols.map((c) => {
                const v = value(r, c);
                return (
                  <td
                    key={c}
                    className="tnum rounded px-2 py-1.5 text-center font-mono"
                    style={{ background: v == null ? "transparent" : color(v), color: v != null && (v - lo) / (hi - lo || 1) < 0.35 ? "#0b1220" : "#f3f6fc" }}
                    title={v == null ? "n/a" : format(v)}
                  >
                    {v == null ? "n/a" : format(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------------------
// Horizontal bar chart (magnitude), value at the bar tip.
// ---------------------------------------------------------------------------------------

export function HBarChart({
  bars,
  format,
  ariaLabel,
}: {
  bars: { key: string; label: string; value: number; color: string; note?: string }[];
  format: (v: number) => string;
  ariaLabel: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const labelW = Math.min(190, Math.max(110, width * 0.34));
  const rowH = 30;
  const max = Math.max(...bars.map((b) => b.value), 1);
  const iw = Math.max(10, width - labelW - 70);
  return (
    <div ref={ref} className="w-full">
      {width > 0 && (
        <svg width={width} height={bars.length * rowH + 4} role="img" aria-label={ariaLabel}>
          {bars.map((b, i) => {
            const w = Math.max(2, (b.value / max) * iw);
            const y = i * rowH + 4;
            return (
              <g key={b.key}>
                <title>{`${b.label}: ${format(b.value)}${b.note ? ` (${b.note})` : ""}`}</title>
                <text x={labelW - 10} y={y + 11} dy="0.32em" textAnchor="end" className="fill-ink-2 text-[11px]">
                  {b.label}
                </text>
                <path d={`M${labelW},${y} h${w - 4} a4,4 0 0 1 4,4 v14 a4,4 0 0 1 -4,4 h${-(w - 4)} z`} fill={b.color} />
                <text x={labelW + w + 8} y={y + 11} dy="0.32em" className="tnum fill-ink text-[11px]">
                  {format(b.value)}
                  {b.note ? <tspan className="fill-ink-3"> {b.note}</tspan> : null}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}
