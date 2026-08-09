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
import { clientKey, type RateLimiter } from "./lib/rate-limit.ts";
import type { Reranker } from "./lib/rerank.ts";
import { search } from "./lib/search.ts";
import { registerSeoRoutes } from "./lib/seo/routes.ts";
import { createShellLoader } from "./lib/seo/shell.ts";
import { PUBLIC_DIR, SITE_URL } from "./lib/seo/site.ts";
import { readSermonTranscript } from "./lib/transcript.ts";

/**
 * The search API, and the server-rendered pages a crawler reads.
 *
 * Built as a factory taking its dependencies rather than importing singletons,
 * so tests can drive the real routes with stub embeddings and no network.
 */

export type AppDeps = {
  prisma: PrismaClient;
  embeddings: EmbeddingsClient;
  reranker?: Reranker | undefined;
  /** Where `transcripts/` lives. Injectable so tests can point at a fixture. */
  dataDir?: string | undefined;
  /** The built SPA. Its `index.html` is the shell the prerenderer writes into. */
  publicDir?: string | undefined;
  /** Absolute origin for canonical and Open Graph URLs. */
  siteUrl?: string | undefined;
  /**
   * Throttles `POST /api/suggestion`. Optional so a test that is not about the
   * limit does not have to think about it; `server.ts` always wires one.
   */
  suggestionLimiter?: RateLimiter | undefined;
};

export function createApp(deps: AppDeps): Hono {
  const app = new Hono();
  const dataDir = deps.dataDir ?? DATA_DIR;

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
      const transcript = await readSermonTranscript(deps.prisma, dataDir, c.req.param("id"));
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

  /**
   * The only write a visitor can make, and so the only route that can grow the
   * database from outside. The limiter is checked before the body is parsed:
   * refusing costs nothing that way, which is the point of refusing.
   */
  app.post("/api/suggestion", async (c) => {
    if (deps.suggestionLimiter && !deps.suggestionLimiter.take(clientKey(c.req.raw.headers))) {
      return c.json({ error: "Muitas sugestões. Tente novamente mais tarde." }, 429);
    }

    const body = await c.req.json().catch(() => ({}));
    const parsed = SuggestionRequestSchema.safeParse(body);
    if (!parsed.success) {
      return c.json({ error: "Sugestão inválida" }, 400);
    }

    await deps.prisma.suggestion.create({ data: { suggestion: parsed.data.suggestion } });
    return c.json({ ok: true });
  });

  /**
   * Last, so nothing here can shadow an API route -- and still ahead of the
   * static middleware and the SPA catch-all, which live in server.ts and would
   * otherwise answer every one of these with the empty shell.
   */
  registerSeoRoutes(app, {
    prisma: deps.prisma,
    dataDir,
    shell: createShellLoader(deps.publicDir ?? PUBLIC_DIR),
    siteUrl: deps.siteUrl ?? SITE_URL,
  });

  return app;
}
