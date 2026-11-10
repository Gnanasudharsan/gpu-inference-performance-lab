import type { NextConfig } from "next";

// Static export: the dashboard is plain files (any static host). Live execution talks to a
// separately deployed API (NEXT_PUBLIC_API_URL), never to code bundled here.
const config: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
  poweredByHeader: false,
};

export default config;
