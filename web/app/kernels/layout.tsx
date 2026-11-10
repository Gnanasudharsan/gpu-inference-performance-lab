import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Kernel analysis",
  description: "Fused kernels for skinny products: launch counts, bandwidth, paired speedups and tuning.",
  alternates: { canonical: "/kernels/" },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
