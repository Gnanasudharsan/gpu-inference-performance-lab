import Link from "next/link";

export const metadata = { title: "Page not found" };

export default function NotFound() {
  return (
    <div className="py-16">
      <div className="font-mono text-[11px] uppercase tracking-[0.14em] text-accent">404</div>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">This page doesn&apos;t exist</h1>
      <p className="mt-3 max-w-xl text-sm leading-relaxed text-ink-2">
        The link may be outdated, or the address mistyped. Every section of the lab is in the navigation.
      </p>
      <div className="mt-6 flex flex-wrap gap-3 text-sm">
        <Link href="/" className="rounded-md bg-accent px-4 py-2 font-medium text-bg">
          Go to the overview
        </Link>
        <Link href="/benchmarks/" className="rounded-md border border-line px-4 py-2 text-ink hover:bg-elevated">
          Browse measurements
        </Link>
      </div>
    </div>
  );
}
