import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Engineering decisions",
  description: "What was chosen, why, and what was rejected.",
  alternates: { canonical: "/decisions/" },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
