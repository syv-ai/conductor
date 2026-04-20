import type { NextConfig } from "next";

const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8765";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
    ];
  },
};

export default nextConfig;
