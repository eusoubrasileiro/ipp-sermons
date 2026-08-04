import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
    coverage: {
      provider: "v8",
      // json-summary is not optional: scripts/quality-gate.mjs reads
      // coverage/coverage-summary.json and treats a missing file as 0%.
      reporter: ["text", "json-summary", "html"],
      reportsDirectory: "./coverage",
      include: ["src/**/*.ts"],
      // server.ts constructs real network and Prisma clients at import time;
      // src/scripts/** are operator entry points. Everything in them worth
      // covering already lives in src/lib/.
      exclude: ["src/server.ts", "src/scripts/**"],
    },
  },
});
