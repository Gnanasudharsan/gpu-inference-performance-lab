import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Latency analysis",
  description: "Synchronous request latency: p50/p95/p99 by batch size and why the tails look the way they do.",
  alternates: { canonical: "/latency/" },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
