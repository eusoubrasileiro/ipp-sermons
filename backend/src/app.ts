import {
  BrowseRequestSchema,
  SearchFiltersSchema,
  SearchRequestSchema,
  SuggestionRequestSchema,
} from "@ipp/shared";
import type { PrismaClient } from "@prisma/client";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { facetCounts } from "./lib/browse/counts.ts";
import { facetTree } from "./lib/browse/facets.ts";
import { type BrowseSort, listSermons } from "./lib/browse/list.ts";
import { filtersFromQuery } from "./lib/browse/query.ts";
import { DATA_DIR } from "./lib/data-dir.ts";
import type { EmbeddingsClient } from "./lib/embeddings.ts";
import type { Reranker } from "./lib/rerank.ts";
import { search } from "./lib/search.ts";
import { readSermonTranscript } from "./lib/transcript.ts";

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
   * The same counts, narrowed to whatever is already filtered.
   *
   * Feeds the "+ filtro" popover, which must never offer a choice that empties
   * the result — so it cannot reuse /api/facets, whose totals are archive-wide.
   */
  app.get("/api/facets/counts", async (c) => {
    const parsed = SearchFiltersSchema.safeParse(filtersFromQuery(c.req.query()));
    if (!parsed.success) {
      return c.json({ error: parsed.error.issues[0]?.message ?? "filtro inválido" }, 400);
    }

    try {
      return c.json(await facetCounts(deps.prisma, parsed.data));
    } catch (err) {
      console.error("[facet-counts] failed", err);
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

    const parsed = BrowseRequestSchema.safeParse({
      filtros: filtersFromQuery(q),
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

  /**
   * A whole sermon, for the reading page.
   *
   * Registered before the SPA fallback in server.ts, or the catch-all would
   * answer it with index.html. Cached for a day because a transcript only
   * changes when the corpus is re-indexed and a release replaces the container
   * anyway; the median file is 40 KB, which is worth not re-sending.
   */
  app.get("/api/sermons/:id/transcript", async (c) => {
    try {
      const transcript = await readSermonTranscript(deps.prisma, DATA_DIR, c.req.param("id"));
      if (!transcript) return c.json({ error: "Sermão não encontrado." }, 404);
      c.header("Cache-Control", "public, max-age=86400");
      return c.json(transcript);
    } catch (err) {
      console.error("[transcript] failed", err);
      return c.json({ error: "Não foi possível carregar a transcrição." }, 500);
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
