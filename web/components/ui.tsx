import type { ReactNode } from "react";
import { VERDICT_STYLE } from "./charts";
import type { Verdict } from "@/lib/types";

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <header className="mb-8">
      <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-accent">{eyebrow}</div>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">{title}</h1>
      {children && <div className="mt-3 max-w-3xl text-sm leading-relaxed text-ink-2">{children}</div>}
    </header>
  );
}

export function Section({ title, description, children }: { title: string; description?: ReactNode; children: ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="text-base font-semibold text-ink">{title}</h2>
      {description && <p className="mt-1 max-w-3xl text-sm leading-relaxed text-ink-2">{description}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-xl border border-line bg-surface p-4 sm:p-5 ${className}`}>{children}</div>;
}

export type Evidence = "measured" | "derived" | "theoretical" | "live" | "pending";

const EVIDENCE_STYLE: Record<Evidence, string> = {
  measured: "border-line text-ink-2",
  derived: "border-line text-ink-2",
  theoretical: "border-warn/40 text-warn",
  live: "border-good/50 text-good",
  pending: "border-line text-ink-3",
};

export function EvidenceTag({ kind }: { kind: Evidence }) {
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-px font-mono text-[10px] uppercase tracking-wider ${EVIDENCE_STYLE[kind]}`}>
      {kind}
    </span>
  );
}

export function StatTile({
  label,
  value,
  unit,
  detail,
  evidence,
}: {
  label: string;
  value: string;
  unit?: string;
  detail?: ReactNode;
  evidence: Evidence;
}) {
  return (
    <Card className="flex flex-col gap-2">
      <div className="flex items-start justify-between gap-2">
        <div className="text-xs text-ink-2">{label}</div>
        <EvidenceTag kind={evidence} />
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="text-3xl font-semibold tracking-tight text-ink">{value}</span>
        {unit && <span className="text-sm text-ink-2">{unit}</span>}
      </div>
      {detail && <div className="text-xs leading-relaxed text-ink-3">{detail}</div>}
    </Card>
  );
}

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  const v = VERDICT_STYLE[verdict];
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap text-xs text-ink-2">
      <span aria-hidden="true" style={{ color: v.color }}>
        {v.icon}
      </span>
      {verdict}
    </span>
  );
}

export function Skeleton({ className = "h-24" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}

export function LoadingPage() {
  return (
    <div aria-busy="true" aria-label="Loading measurements">
      <Skeleton className="mb-3 h-4 w-32" />
      <Skeleton className="mb-8 h-8 w-80" />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-32" />
        ))}
      </div>
      <Skeleton className="mt-8 h-72" />
    </div>
  );
}

export function ErrorState({ title, message, onRetry }: { title: string; message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-xl border border-bad/40 bg-bad/5 p-5">
      <div className="text-sm font-medium text-ink">
        <span aria-hidden="true" className="mr-1.5 text-bad">
          ▼
        </span>
        {title}
      </div>
      <p className="mt-1 text-sm text-ink-2">{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="mt-3 rounded-md border border-line px-3 py-1 text-sm text-ink hover:bg-elevated">
          Retry
        </button>
      )}
    </div>
  );
}

export function Code({ children }: { children: ReactNode }) {
  return <code className="rounded bg-elevated px-1.5 py-0.5 font-mono text-[12px] text-ink">{children}</code>;
}

export function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="overflow-x-auto rounded-lg border border-line bg-bg p-4 font-mono text-[12px] leading-relaxed text-ink">
      <code>{children}</code>
    </pre>
  );
}

export function Table({ head, rows, caption }: { head: ReactNode[]; rows: ReactNode[][]; caption?: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-line text-ink-3">
            {head.map((h, i) => (
              <th key={i} scope="col" className="whitespace-nowrap px-2 py-2 font-normal">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tnum">
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line/60 last:border-0 hover:bg-elevated/50">
              {r.map((c, j) => (
                <td key={j} className="whitespace-nowrap px-2 py-1.5 text-ink-2 first:text-ink">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
