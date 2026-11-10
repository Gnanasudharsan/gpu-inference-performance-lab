import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Benchmark explorer",
  description: "Every recorded measurement, filterable by device, precision, model, implementation and batch size.",
  alternates: { canonical: "/benchmarks/" },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
