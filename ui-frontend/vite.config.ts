import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Dev proxy lets `npm run dev` talk to a running `fastaiagent ui start` on 7842.
const API_PROXY = process.env.VITE_API_PROXY ?? "http://127.0.0.1:7842";

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
