import type { MetadataRoute } from "next";

export const dynamic = "force-static";

const SITE = "https://gpu-inference-performance-lab.vercel.app";
const ROUTES = ["", "live", "benchmarks", "latency", "serving", "kernels", "decisions", "reproducibility", "architecture"];

export default function sitemap(): MetadataRoute.Sitemap {
  return ROUTES.map((r) => ({ url: r ? `${SITE}/${r}/` : `${SITE}/`, changeFrequency: "weekly", priority: r ? 0.6 : 1 }));
}
