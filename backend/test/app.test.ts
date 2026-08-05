import type { PrismaClient } from "@prisma/client";
import { describe, expect, it, vi } from "vitest";
import { createApp } from "../src/app.ts";
import type { EmbeddingsClient } from "../src/lib/embeddings.ts";
import type { Reranker } from "../src/lib/rerank.ts";

/**
 * Route-level tests driving the real Hono app with stubbed dependencies, so
 * validation, error mapping and the rerank fallback are exercised end to end
 * without a database or any network call.
 */

const chunkRow = (i: number, date = "2020-05-03") => ({
  id: `c${i}`,
  sermon_id: `s${i}`,
  chunk_index: i,
  content: `trecho ${i}`,
  score: 0.03 - i * 0.001,
  title: `Sermão ${i}`,
  artist: "Reverendo Bruno Melo",
  date: new Date(`${date}T00:00:00Z`),
  duration_str: "45:49",
  sc_suffix_url: `sermao-${i}`,
  sp_suffix_url: `spid${i}`,
});

const stubEmbeddings = (): EmbeddingsClient => ({
  embed: vi.fn(async (inputs: string[]) => inputs.map(() => [0.1, 0.2, 0.3])),
});

const stubPrisma = (rows: unknown[]) =>
  ({
    $queryRaw: vi.fn(async () => rows),
    sermon: { count: vi.fn(async () => 455) },
    sermonChunk: { count: vi.fn(async () => 19953) },
    suggestion: { create: vi.fn(async () => ({})) },
  }) as unknown as PrismaClient;

type SearchBody = {
  query: string;
  reranked: boolean;
  tookMs: number;
  results: {
    chunkIndex: number;
    score: number;
    soundcloudUrl: string | null;
    spotifyUrl: string | null;
    date: string;
  }[];
};

const readSearch = async (res: Response): Promise<SearchBody> => (await res.json()) as SearchBody;

const post = (app: ReturnType<typeof createApp>, path: string, body: unknown) =>
  app.request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

describe("GET /api/health", () => {
  it("reports corpus counts", async () => {
    const app = createApp({ prisma: stubPrisma([]), embeddings: stubEmbeddings() });
    const res = await app.request("/api/health");

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ status: "ok", sermons: 455, chunks: 19953 });
  });
});

describe("POST /api/search", () => {
  it("returns results and builds playable urls", async () => {
    const app = createApp({
      prisma: stubPrisma([chunkRow(0, "2023-03-19")]),
      embeddings: stubEmbeddings(),
    });
    const res = await post(app, "/api/search", { query: "graça" });

    expect(res.status).toBe(200);
    const body = await readSearch(res);
    expect(body.results).toHaveLength(1);
    // The channel segment is mandatory: sc_suffix_url is only the track slug.
    expect(body.results[0]?.soundcloudUrl).toBe("https://soundcloud.com/ipperegrinos/sermao-0");
    expect(body.results[0]?.spotifyUrl).toBe("https://open.spotify.com/episode/spid0");
    expect(body.results[0]?.date).toBe("2023-03-19");
    expect(body.reranked).toBe(false);
  });

  it("withholds the Spotify link for pre-2022 sermons but keeps SoundCloud", async () => {
    // Those episodes were retired upstream; see SPOTIFY_LINKS_ALIVE_FROM.
    const app = createApp({
      prisma: stubPrisma([chunkRow(0, "2021-12-31")]),
      embeddings: stubEmbeddings(),
    });
    const body = await readSearch(await post(app, "/api/search", { query: "graça" }));

    expect(body.results[0]?.spotifyUrl).toBeNull();
    expect(body.results[0]?.soundcloudUrl).toBe("https://soundcloud.com/ipperegrinos/sermao-0");
  });

  it("rejects a too-short query with 400", async () => {
    const app = createApp({ prisma: stubPrisma([]), embeddings: stubEmbeddings() });
    const res = await post(app, "/api/search", { query: "a" });

    expect(res.status).toBe(400);
  });

  it("rejects a malformed body with 400", async () => {
    const app = createApp({ prisma: stubPrisma([]), embeddings: stubEmbeddings() });
    const res = await post(app, "/api/search", {});

    expect(res.status).toBe(400);
  });

  it("honours the limit", async () => {
    const rows = [chunkRow(0), chunkRow(1), chunkRow(2)];
    const app = createApp({ prisma: stubPrisma(rows), embeddings: stubEmbeddings() });
    const res = await post(app, "/api/search", { query: "graça", limit: 2 });

    expect((await readSearch(res)).results).toHaveLength(2);
  });

  it("returns 500 rather than leaking an internal error", async () => {
    const prisma = {
      $queryRaw: vi.fn(async () => {
        throw new Error("connection refused at 10.0.0.5");
      }),
    } as unknown as PrismaClient;
    const app = createApp({ prisma, embeddings: stubEmbeddings() });
    const res = await post(app, "/api/search", { query: "graça" });

    expect(res.status).toBe(500);
    expect(JSON.stringify(await res.json())).not.toContain("10.0.0.5");
  });

  it("applies the reranker's order when it succeeds", async () => {
    const rows = [chunkRow(0), chunkRow(1)];
    const reranker: Reranker = {
      // Reverse the RRF order to prove the reranker is what decided it.
      rerank: vi.fn(async () => [
        { index: 1, score: 0.99 },
        { index: 0, score: 0.11 },
      ]),
    };
    const app = createApp({ prisma: stubPrisma(rows), embeddings: stubEmbeddings(), reranker });
    const body = await readSearch(await post(app, "/api/search", { query: "graça", limit: 2 }));

    expect(body.reranked).toBe(true);
    expect(body.results[0]?.chunkIndex).toBe(1);
    expect(body.results[0]?.score).toBe(0.99);
  });

  it("falls back to RRF order when the reranker fails", async () => {
    // The load-bearing behaviour: a reranker outage degrades ranking, never
    // the availability of search.
    const rows = [chunkRow(0), chunkRow(1)];
    const reranker: Reranker = { rerank: vi.fn(async () => null) };
    const app = createApp({ prisma: stubPrisma(rows), embeddings: stubEmbeddings(), reranker });
    const res = await post(app, "/api/search", { query: "graça", limit: 2 });
    const body = await readSearch(res);

    expect(res.status).toBe(200);
    expect(body.reranked).toBe(false);
    expect(body.results[0]?.chunkIndex).toBe(0);
  });
});

describe("POST /api/suggestion", () => {
  it("stores a valid suggestion", async () => {
    const prisma = stubPrisma([]);
    const app = createApp({ prisma, embeddings: stubEmbeddings() });
    const res = await post(app, "/api/suggestion", { suggestion: "falta o sermão de Tito 2" });

    expect(res.status).toBe(200);
    expect(prisma.suggestion.create).toHaveBeenCalled();
  });

  it("rejects an empty suggestion", async () => {
    const app = createApp({ prisma: stubPrisma([]), embeddings: stubEmbeddings() });
    expect((await post(app, "/api/suggestion", { suggestion: " " })).status).toBe(400);
  });
});

describe("POST /api/search — filtros", () => {
  it("passes the facets through as bound parameters", async () => {
    const prisma = stubPrisma([chunkRow(1)]);
    const app = createApp({ prisma, embeddings: stubEmbeddings() });

    const res = await app.request("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: "casamento",
        filtros: { livros: ["efesios"], capitulo: 5, pregadores: ["Reverendo Bruno Melo"] },
      }),
    });

    expect(res.status).toBe(200);
    // The facet values must arrive as bound parameters, never spliced into SQL.
    const bound = JSON.stringify(
      (prisma.$queryRaw as unknown as { mock: { calls: unknown[] } }).mock.calls,
    );
    expect(bound).toContain("efesios");
    expect(bound).toContain("Reverendo Bruno Melo");
  });

  it("rejects a chapter that cannot exist", async () => {
    const app = createApp({ prisma: stubPrisma([]), embeddings: stubEmbeddings() });
    const res = await app.request("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "casamento", filtros: { capitulo: 999 } }),
    });
    expect(res.status).toBe(400);
  });

  it("searches unfiltered when no facets are sent", async () => {
    const app = createApp({ prisma: stubPrisma([chunkRow(1)]), embeddings: stubEmbeddings() });
    const res = await app.request("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "casamento" }),
    });
    expect(res.status).toBe(200);
  });
});

describe("GET /api/facets", () => {
  const stubFacets = () =>
    ({
      $queryRaw: vi.fn(async () => []),
      sermon: { count: vi.fn(async () => 0), findMany: vi.fn(async () => []) },
      sermonChunk: { count: vi.fn(async () => 0) },
      suggestion: { create: vi.fn(async () => ({})) },
    }) as unknown as PrismaClient;

  it("returns every facet family, so the client fetches the tree once", async () => {
    const app = createApp({ prisma: stubFacets(), embeddings: stubEmbeddings() });
    const res = await app.request("/api/facets");

    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown[]>;
    for (const family of ["livros", "series", "pregadores", "datas", "tipos", "temas"]) {
      expect(body).toHaveProperty(family);
    }
  });

  it("does not leak an internal error to the browser", async () => {
    const prisma = {
      $queryRaw: vi.fn(async () => {
        throw new Error("connection refused at 10.0.0.5");
      }),
    } as unknown as PrismaClient;

    const res = await createApp({ prisma, embeddings: stubEmbeddings() }).request("/api/facets");
    expect(res.status).toBe(500);
    expect(JSON.stringify(await res.json())).not.toContain("10.0.0.5");
  });
});

describe("GET /api/sermons", () => {
  const stubBrowse = () =>
    ({
      sermon: { count: vi.fn(async () => 3), findMany: vi.fn(async () => []) },
      $queryRaw: vi.fn(async () => []),
    }) as unknown as PrismaClient;

  it("browses with no query at all", async () => {
    // The point of this route: SearchRequestSchema needs two characters of
    // query, and browsing by book has none.
    const app = createApp({ prisma: stubBrowse(), embeddings: stubEmbeddings() });
    const res = await app.request("/api/sermons?livros=efesios&capitulo=5");

    expect(res.status).toBe(200);
    expect((await res.json()) as { total: number }).toMatchObject({ total: 3 });
  });

  it("reads a comma-separated facet as a list", async () => {
    const prisma = stubBrowse();
    const app = createApp({ prisma, embeddings: stubEmbeddings() });
    await app.request("/api/sermons?pregadores=Reverendo%20Bruno%20Melo,Pastor%20Lucas%20Antunes");

    const where = (prisma.sermon.count as unknown as { mock: { calls: unknown[][] } }).mock
      .calls[0]?.[0];
    expect(JSON.stringify(where)).toContain("Lucas Antunes");
  });

  it("rejects an unknown sort order", async () => {
    const app = createApp({ prisma: stubBrowse(), embeddings: stubEmbeddings() });
    expect((await app.request("/api/sermons?ordenar=aleatorio")).status).toBe(400);
  });

  it("defaults to the newest first", async () => {
    const prisma = stubBrowse();
    await createApp({ prisma, embeddings: stubEmbeddings() }).request("/api/sermons");

    const args = (prisma.sermon.findMany as unknown as { mock: { calls: unknown[][] } }).mock
      .calls[0]?.[0];
    expect(JSON.stringify(args)).toContain('"date":"desc"');
  });
});

describe("GET /api/facets/counts", () => {
  const sermonRow = (over: Record<string, unknown> = {}) => ({
    artist: "Reverendo Bruno Melo",
    serviceType: "culto",
    seriesSlug: null,
    date: new Date("2024-03-17T00:00:00Z"),
    scriptures: [{ bookSlug: "efesios", chapter: 5 }],
    topics: [],
    ...over,
  });

  const stubCounts = (rows: unknown[]) =>
    ({ sermon: { findMany: vi.fn(async () => rows) } }) as unknown as PrismaClient;

  it("narrows the counts to the filters already chosen", async () => {
    const prisma = stubCounts([sermonRow(), sermonRow({ artist: "Pastor Lucas Antunes" })]);
    const app = createApp({ prisma, embeddings: stubEmbeddings() });

    const res = await app.request("/api/facets/counts?pregadores=Reverendo%20Bruno%20Melo");
    const body = (await res.json()) as { total: number; livros: Record<string, number> };

    expect(res.status).toBe(200);
    expect(body.total).toBe(1);
    expect(body.livros).toEqual({ efesios: 1 });
  });

  it("rejects a chapter outside the range any book has", async () => {
    const app = createApp({ prisma: stubCounts([]), embeddings: stubEmbeddings() });
    expect((await app.request("/api/facets/counts?capitulo=999")).status).toBe(400);
  });

  it("keeps the database error off the wire", async () => {
    const prisma = {
      sermon: {
        findMany: vi.fn(async () => {
          throw new Error("connect ECONNREFUSED 10.0.0.5:5432");
        }),
      },
    } as unknown as PrismaClient;

    const res = await createApp({ prisma, embeddings: stubEmbeddings() }).request(
      "/api/facets/counts",
    );
    expect(res.status).toBe(500);
    expect(JSON.stringify(await res.json())).not.toContain("10.0.0.5");
  });
});
