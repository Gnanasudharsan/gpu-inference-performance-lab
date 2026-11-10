import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Reproducibility",
  description: "Environment, commands, experiment configurations and provenance.",
  alternates: { canonical: "/reproducibility/" },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
