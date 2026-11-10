"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/live/", label: "Live execution" },
  { href: "/benchmarks/", label: "Benchmark explorer" },
  { href: "/latency/", label: "Latency analysis" },
  { href: "/serving/", label: "Serving" },
  { href: "/kernels/", label: "Kernel analysis" },
  { href: "/decisions/", label: "Engineering decisions" },
  { href: "/reproducibility/", label: "Reproducibility" },
  { href: "/architecture/", label: "Architecture" },
];

const REPO = "https://github.com/Gnanasudharsan/gpu-inference-performance-lab";

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const norm = (p: string) => (p.endsWith("/") ? p : `${p}/`);
  const nav = (
    <nav aria-label="Sections">
      <ul className="space-y-0.5">
        {NAV.map((n) => {
          const active = norm(path ?? "/") === norm(n.href);
          return (
            <li key={n.href}>
              <Link
                href={n.href}
                onClick={() => setOpen(false)}
                aria-current={active ? "page" : undefined}
                className={`block rounded-md px-3 py-1.5 text-sm transition-colors duration-150 ${active ? "bg-elevated text-ink" : "text-ink-2 hover:bg-elevated/60 hover:text-ink"}`}
              >
                {active && <span aria-hidden="true" className="mr-2 inline-block h-1.5 w-1.5 rounded-full bg-accent align-middle" />}
                {n.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
  const brand = (
    <Link href="/" className="block">
      <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-accent">gpulab</div>
      <div className="text-sm font-semibold text-ink">GPU Inference Performance Lab</div>
    </Link>
  );
  return (
    <div className="min-h-screen lg:flex">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-elevated focus:px-3 focus:py-2">
        Skip to content
      </a>
      <aside className="hidden w-64 shrink-0 border-r border-line bg-surface/60 px-4 py-6 lg:sticky lg:top-0 lg:block lg:h-screen lg:overflow-y-auto">
        <div className="mb-8 px-3">{brand}</div>
        {nav}
        <div className="mt-8 space-y-2 border-t border-line px-3 pt-4 text-xs text-ink-3">
          <a className="block hover:text-ink" href={REPO}>
            Source on GitHub ↗
          </a>
          <p>Every number links to the device and commit that produced it.</p>
        </div>
      </aside>
      <div className="sticky top-0 z-30 flex items-center justify-between border-b border-line bg-bg/95 px-4 py-3 backdrop-blur lg:hidden">
        {brand}
        <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-controls="mobile-nav" className="rounded-md border border-line px-3 py-1.5 text-sm text-ink">
          {open ? "Close" : "Menu"}
        </button>
      </div>
      {open && (
        <div id="mobile-nav" className="border-b border-line bg-surface px-4 py-3 lg:hidden">
          {nav}
        </div>
      )}
      <main id="main" className="min-w-0 flex-1 px-4 py-8 sm:px-8 lg:px-12 lg:py-10">
        <div className="mx-auto max-w-6xl">{children}</div>
      </main>
    </div>
  );
}
