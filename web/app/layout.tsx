import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Shell } from "@/components/Shell";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains", display: "swap" });

const SITE = "https://gpu-inference-performance-lab.vercel.app";
const DESCRIPTION =
  "Reproducible small-batch GPU inference optimization: measured latency, fused kernels, paired speedups with confidence intervals, and latency-aware serving.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE),
  title: { default: "GPU Inference Performance Lab", template: "%s · GPU Inference Performance Lab" },
  description: DESCRIPTION,
  applicationName: "GPU Inference Performance Lab",
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    url: SITE,
    siteName: "GPU Inference Performance Lab",
    title: "GPU Inference Performance Lab",
    description: DESCRIPTION,
    images: [{ url: "/og.png", width: 1200, height: 630, alt: "GPU Inference Performance Lab: measured results" }],
  },
  twitter: { card: "summary_large_image", title: "GPU Inference Performance Lab", description: DESCRIPTION, images: ["/og.png"] },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = { themeColor: "#0b1220", colorScheme: "dark" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`}>
      <body className="font-sans">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
