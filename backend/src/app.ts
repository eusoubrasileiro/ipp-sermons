import { BrowseRequestSchema, SearchRequestSchema, SuggestionRequestSchema } from "@ipp/shared";
import type { PrismaClient } from "@prisma/client";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { facetTree } from "./lib/browse/facets.ts";
import { type BrowseSort, listSermons } from "./lib/browse/list.ts";
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

  /**
   * The whole index tree with counts, for the browse pages.
   *
   * One request rather than six: the tree is a few kilobytes over 456 sermons
   * and changes only when the corpus is re-indexed, so the client fetches it
   * once and every index page renders from memory.
   */
  app.get("/api/facets", async (c) => {
    try {
      return c.json(await facetTree(deps.prisma));
    } catch (err) {
      console.error("[facets] failed", err);
      return c.json({ error: "Não foi possível carregar os índices." }, 500);
    }
  });

  /**
   * A filtered listing with no query text.
   *
   * Separate from POST /api/search because SearchRequestSchema requires at
   * least two characters of query -- browsing by book or preacher has none,
   * and relaxing that would weaken the contract the search route relies on.
   */
  app.get("/api/sermons", async (c) => {
    const q = c.req.query();
    const asArray = (v: string | undefined) => (v ? v.split(",").filter(Boolean) : undefined);

    const parsed = BrowseRequestSchema.safeParse({
      filtros: {
        pregadores: asArray(q.pregadores),
        tipos: asArray(q.tipos),
        series: asArray(q.series),
        livros: asArray(q.livros),
        temas: asArray(q.temas),
        capitulo: q.capitulo ? Number.parseInt(q.capitulo, 10) : undefined,
        de: q.de,
        ate: q.ate,
      },
      ordenar: q.ordenar,
      pagina: q.pagina ? Number.parseInt(q.pagina, 10) : undefined,
    });

    if (!parsed.success) {
      return c.json({ error: parsed.error.issues[0]?.message ?? "filtro inválido" }, 400);
    }

    try {
      const { filtros, ordenar, pagina } = parsed.data;
      const result = await listSermons(deps.prisma, filtros ?? {}, ordenar as BrowseSort, pagina);
      return c.json({ ...result, pagina });
    } catch (err) {
      console.error("[sermons] failed", err);
      return c.json({ error: "Não foi possível listar os sermões." }, 500);
    }
  });

  app.post("/api/search", async (c) => {
    const body = await c.req.json().catch(() => ({}));
    const parsed = SearchRequestSchema.safeParse(body);
    if (!parsed.success) {
      return c.json({ error: parsed.error.issues[0]?.message ?? "consulta inválida" }, 400);
    }

    const { query, limit, filtros } = parsed.data;
    const started = Date.now();

    try {
      const { results, reranked } = await search(deps, query, limit, filtros);
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
