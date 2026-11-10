import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Serving",
  description: "Open-loop serving under deadlines: goodput, miss rate and tail latency for direct, static and dynamic batching.",
  alternates: { canonical: "/serving/" },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
