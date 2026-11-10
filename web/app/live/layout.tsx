import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Live execution",
  description: "Run a bounded, paired measurement on a real gpulab backend.",
  alternates: { canonical: "/live/" },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
