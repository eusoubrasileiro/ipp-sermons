import react from "@vitejs/plugin-react-swc";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Build straight into the backend's static dir: one container serves both,
  // so there is no CORS and no second origin in production.
  build: { outDir: "../backend/public", emptyOutDir: true },
  server: {
    proxy: { "/api": "http://localhost:3311" },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.test.tsx"],
  },
});
