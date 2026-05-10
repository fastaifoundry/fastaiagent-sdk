import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev proxy lets `npm run dev` talk to a running `fastaiagent ui start` on 7842.
//
// security_review_1.md M13: validate that the configured proxy target
// is loopback before binding the dev server. Otherwise a dev would
// silently start sending session cookies + dev creds to whatever host
// ``VITE_API_PROXY`` happens to point at — easy to misconfigure (typo
// in `.env.local`, copy-paste from a teammate's prod setup) and easy
// to overlook because there is no startup banner saying where /api
// is going.
const _DEV_LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "localhost"]);

function _validateApiProxy(rawTarget: string): string {
  let parsed: URL;
  try {
    parsed = new URL(rawTarget);
  } catch {
    throw new Error(
      `VITE_API_PROXY must be an absolute URL with explicit scheme; got ${JSON.stringify(rawTarget)}. ` +
        `Example: http://127.0.0.1:7842`
    );
  }
  const host = parsed.hostname.toLowerCase();
  if (!_DEV_LOOPBACK_HOSTS.has(host) && !host.endsWith(".localhost")) {
    throw new Error(
      `VITE_API_PROXY must point at a loopback host (got ${parsed.hostname}). ` +
        `The Local UI is single-user and runs on 127.0.0.1 only. ` +
        `If you really need a remote dev backend, set up a TLS-terminating ` +
        `tunnel pointing at 127.0.0.1 instead.`
    );
  }
  return rawTarget;
}

const API_PROXY = _validateApiProxy(
  process.env.VITE_API_PROXY ?? "http://127.0.0.1:7842"
);

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    // The Python wheel ships whatever lands here.
    outDir: path.resolve(__dirname, "../fastaiagent/ui/static"),
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_PROXY,
        changeOrigin: true,
      },
    },
  },
});
