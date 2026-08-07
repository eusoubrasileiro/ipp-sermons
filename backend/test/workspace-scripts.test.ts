import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * The pnpm scripts are an interface, and nothing else checks them.
 *
 * A corpus update runs eight of them in an order where two of the steps
 * silently destroy the previous one's output if reversed. Once that order lives
 * in a script rather than in someone's memory, the names it calls and the
 * environment they load stop being cosmetic: a script that exists only inside
 * `backend/` is a stage the orchestrator cannot invoke, and a script that does
 * not load `.env` dies on a missing key halfway through a paid run.
 */
const ROOT = join(import.meta.dirname, "../..");

function scripts(pkg: string): Record<string, string> {
  return JSON.parse(readFileSync(join(ROOT, pkg, "package.json"), "utf8")).scripts;
}

/** The facet pipeline, in the order a corpus update runs it. */
const PIPELINE = [
  "verify:corpus",
  "derive:facets",
  "extract:scripture",
  "verify:facets",
  "label:topics",
  "index",
  "index:facets",
  "eval",
];

describe("workspace scripts", () => {
  it("reaches every pipeline stage from the workspace root", () => {
    const root = scripts(".");
    expect(PIPELINE.filter((name) => !(name in root))).toEqual([]);
  });

  it("keeps propose:taxonomy out of reach of the root", () => {
    // It rewrites taxonomy.csv from scratch, which orphans every row
    // label-topics.ts ever wrote. Leaving it backend-only is a free structural
    // guard against it ever joining the routine loop.
    expect(scripts(".")).not.toHaveProperty("propose:taxonomy");
    expect(scripts("backend")).toHaveProperty("propose:taxonomy");
  });

  it("loads .env in every backend script that runs one of src/scripts", () => {
    // Half of them did and half did not, which is why `pnpm eval` died with
    // "OPENROUTER_API_KEY is not set" and why run.sh had to source .env itself
    // before it could reach the indexer.
    const missing = Object.entries(scripts("backend"))
      .filter(([, cmd]) => cmd.includes("src/scripts/"))
      .filter(([, cmd]) => !cmd.includes("--env-file-if-exists=../.env"))
      .map(([name]) => name);

    expect(missing).toEqual([]);
  });

  it("names every migration NNN_*.sql", () => {
    // Both the production `migrate` sidecar (`for f in /sql/*.sql`) and
    // db-push.sh (`[0-9][0-9][0-9]_*.sql`) apply these in filename order. A file
    // that sorts wrong is a migration applied at the wrong time.
    const sql = readdirSync(join(ROOT, "backend/prisma/sql"));
    expect(sql.filter((f) => !/^\d{3}_[a-z0-9_-]+\.sql$/.test(f))).toEqual([]);
    expect(sql.length).toBeGreaterThan(0);
  });
});
