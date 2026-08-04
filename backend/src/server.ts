import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import pkg from "@prisma/client";

// @prisma/client is CommonJS: `import { PrismaClient }` resolves under the dev
// loader but not in the compiled ESM build, where named-export detection
// misses it. Destructure from the default export instead.
const { PrismaClient } = pkg;

import { createApp } from "./app.ts";
import { createOpenRouterEmbeddings } from "./lib/embeddings.ts";
import { createOpenRouterReranker } from "./lib/rerank.ts";

/**
 * Production entrypoint: wires the real dependencies and serves the built
 * frontend from the same origin, so there is one container and no CORS in prod.
 */

const port = Number.parseInt(process.env.PORT ?? "3000", 10);

const apiKey = process.env.OPENROUTER_API_KEY;
if (!apiKey) throw new Error("OPENROUTER_API_KEY is not set");

const prisma = new PrismaClient();
const embeddings = createOpenRouterEmbeddings({ apiKey });

// Reranking is on unless explicitly disabled. It is a precision win, but the
// search must keep working without it -- see lib/rerank.ts.
const rerankDisabled = process.env.RERANK_DISABLED === "1";
const reranker = rerankDisabled
  ? undefined
  : createOpenRouterReranker({
      apiKey,
      onError: (err) => console.warn("[rerank] degraded to RRF order:", err),
    });

const app = createApp({ prisma, embeddings, reranker });

app.use("/*", serveStatic({ root: "./public" }));
// SPA fallback: any unmatched GET returns the shell so client routing works.
app.get("*", serveStatic({ path: "./public/index.html" }));

serve({ fetch: app.fetch, port }, (info) => {
  console.log(`ipp-sermons listening on :${info.port} (rerank ${rerankDisabled ? "off" : "on"})`);
});

const shutdown = async (): Promise<void> => {
  await prisma.$disconnect();
  process.exit(0);
};
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
