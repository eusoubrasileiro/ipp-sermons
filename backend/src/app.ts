import { SearchRequestSchema, SuggestionRequestSchema } from "@ipp/shared";
import type { PrismaClient } from "@prisma/client";
import { Hono } from "hono";
import { cors } from "hono/cors";
import type { EmbeddingsClient } from "./lib/embeddings.ts";
import type { Reranker } from "./lib/rerank.ts";
import { search } from "./lib/search.ts";

/**
 * The search API.
 *
 * Built as a factory taking its dependencies rather than importing singletons,
 * so tests can drive the real routes with stub embeddings and no network.
 */

export type AppDeps = {
  prisma: PrismaClient;
  embeddings: EmbeddingsClient;
  reranker?: Reranker | undefined;
};

export function createApp(deps: AppDeps): Hono {
  const app = new Hono();

  app.use("/api/*", cors());

  app.get("/api/health", async (c) => {
    const [sermons, chunks] = await Promise.all([
      deps.prisma.sermon.count(),
      deps.prisma.sermonChunk.count(),
    ]);
    return c.json({ status: "ok" as const, sermons, chunks });
  });

  app.post("/api/search", async (c) => {
    const body = await c.req.json().catch(() => ({}));
    const parsed = SearchRequestSchema.safeParse(body);
    if (!parsed.success) {
      return c.json({ error: parsed.error.issues[0]?.message ?? "consulta inválida" }, 400);
    }

    const { query, limit } = parsed.data;
    const started = Date.now();

    try {
      const { results, reranked } = await search(deps, query, limit);
      return c.json({ query, results, reranked, tookMs: Date.now() - started });
    } catch (err) {
      console.error("[search] failed", err);
      return c.json({ error: "A busca falhou. Tente novamente." }, 500);
    }
  });

  app.post("/api/suggestion", async (c) => {
    const body = await c.req.json().catch(() => ({}));
    const parsed = SuggestionRequestSchema.safeParse(body);
    if (!parsed.success) {
      return c.json({ error: "Sugestão inválida" }, 400);
    }

    await deps.prisma.suggestion.create({ data: { suggestion: parsed.data.suggestion } });
    return c.json({ ok: true });
  });

  return app;
}
