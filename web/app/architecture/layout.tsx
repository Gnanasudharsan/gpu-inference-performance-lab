import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Architecture",
  description: "How the measurement package, dashboard and live API fit together.",
  alternates: { canonical: "/architecture/" },
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
