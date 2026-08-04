import { readFile } from "node:fs/promises";
import { join } from "node:path";
import pkg from "@prisma/client";

// @prisma/client is CommonJS: `import { PrismaClient }` resolves under the dev
// loader but not in the compiled ESM build. Destructure from the default.
const { PrismaClient } = pkg;

import { createOpenRouterEmbeddings } from "../lib/embeddings.ts";
import { createOpenRouterReranker } from "../lib/rerank.ts";
import { search } from "../lib/search.ts";

/**
 * Runs the golden query set against the real database and embedding API.
 *
 * This is the check that the API-only rewrite retrieves as well as the GPU
 * system it replaces. Unit tests prove the plumbing; only this proves the
 * search is any good.
 *
 *   pnpm eval            # RRF only
 *   RERANK=1 pnpm eval   # with the cross-encoder
 */

type GoldenQuery = {
  query: string;
  expect: string[];
  why: string;
  minResults?: number;
};

type GoldenSet = {
  recallAt: number;
  queries: GoldenQuery[];
};

async function main(): Promise<void> {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is not set");

  const dir = process.env.GOLDEN_DIR ?? join(import.meta.dirname, "../../test/golden");
  const golden = JSON.parse(await readFile(join(dir, "queries.json"), "utf8")) as GoldenSet;

  const prisma = new PrismaClient();
  const embeddings = createOpenRouterEmbeddings({ apiKey });
  const useRerank = process.env.RERANK === "1";
  const reranker = useRerank ? createOpenRouterReranker({ apiKey }) : undefined;

  console.log(`golden set: ${golden.queries.length} queries, recall@${golden.recallAt}`);
  console.log(`rerank: ${useRerank ? "on" : "off"}\n`);

  let passed = 0;
  const failures: string[] = [];

  for (const gq of golden.queries) {
    const { results } = await search({ prisma, embeddings, reranker }, gq.query, golden.recallAt);
    const foundIds = new Set(results.map((r) => r.id));

    const missing = gq.expect.filter((id) => !foundIds.has(id));
    const tooFew = (gq.minResults ?? 0) > results.length;
    const ok = missing.length === 0 && !tooFew;

    if (ok) passed++;
    else {
      failures.push(
        `${gq.query}: ${missing.length > 0 ? `missing ${missing.join(", ")}` : `only ${results.length} results`}`,
      );
    }

    const mark = ok ? "PASS" : "FAIL";
    console.log(`  [${mark}] ${gq.query}`);
    console.log(`         ${results.length} results; ${gq.why}`);
    if (!ok && results.length > 0) {
      for (const r of results.slice(0, 3)) {
        console.log(`         got: ${r.id} ${r.title.slice(0, 50)}`);
      }
    }
  }

  const recall = (passed / golden.queries.length) * 100;
  console.log(
    `\nrecall@${golden.recallAt}: ${passed}/${golden.queries.length} (${recall.toFixed(0)}%)`,
  );
  for (const f of failures) console.log(`  failed: ${f}`);

  await prisma.$disconnect();
  if (failures.length > 0) process.exit(1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
