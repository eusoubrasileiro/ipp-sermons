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

const chunkRow = (i: number) => ({
  id: `c${i}`,
  sermon_id: `s${i}`,
  chunk_index: i,
  content: `trecho ${i}`,
  score: 0.03 - i * 0.001,
  title: `Sermão ${i}`,
  artist: "Reverendo Bruno Melo",
  date: new Date("2020-05-03T00:00:00Z"),
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
    const app = createApp({ prisma: stubPrisma([chunkRow(0)]), embeddings: stubEmbeddings() });
    const res = await post(app, "/api/search", { query: "graça" });

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.results).toHaveLength(1);
    expect(body.results[0].soundcloudUrl).toBe("https://soundcloud.com/sermao-0");
    expect(body.results[0].spotifyUrl).toBe("https://open.spotify.com/episode/spid0");
    expect(body.results[0].date).toBe("2020-05-03");
    expect(body.reranked).toBe(false);
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

    expect((await res.json()).results).toHaveLength(2);
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
    const body = await (await post(app, "/api/search", { query: "graça", limit: 2 })).json();

    expect(body.reranked).toBe(true);
    expect(body.results[0].chunkIndex).toBe(1);
    expect(body.results[0].score).toBe(0.99);
  });

  it("falls back to RRF order when the reranker fails", async () => {
    // The load-bearing behaviour: a reranker outage degrades ranking, never
    // the availability of search.
    const rows = [chunkRow(0), chunkRow(1)];
    const reranker: Reranker = { rerank: vi.fn(async () => null) };
    const app = createApp({ prisma: stubPrisma(rows), embeddings: stubEmbeddings(), reranker });
    const res = await post(app, "/api/search", { query: "graça", limit: 2 });
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.reranked).toBe(false);
    expect(body.results[0].chunkIndex).toBe(0);
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
