import react from "@vitejs/plugin-react-swc";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Build straight into the backend's static dir: one container serves both,
  // so there is no CORS and no second origin in production.
  build: { outDir: "../backend/public", emptyOutDir: true },
  server: {
    // Override to point the dev frontend at a backend on another port.
    proxy: { "/api": process.env.API_PROXY_TARGET ?? "http://localhost:3311" },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    coverage: {
      provider: "v8",
      // json-summary is not optional: scripts/quality-gate.mjs reads
      // coverage/coverage-summary.json and treats a missing file as 0%.
      reporter: ["text", "json-summary", "html"],
      reportsDirectory: "./coverage",
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/main.tsx", "src/test-setup.ts", "src/**/*.test.{ts,tsx}"],
    },
  },
});
