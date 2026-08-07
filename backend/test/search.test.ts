import type { PrismaClient } from "@prisma/client";
import { describe, expect, it, vi } from "vitest";
import { search } from "../src/lib/search.ts";

/**
 * `limit` counts sermons, because a sermon is what the reader is looking for.
 *
 * `hybrid_search()` ranks transcript chunks and deliberately lets one sermon
 * contribute two of them, so that a card can show a second matching passage.
 * Slicing that list to `limit` spent the page budget on the extra passages: ten
 * results were seven sermons, and the eighth-best sermon was never returned at
 * all. The frontend already counts `groupBySermon(...).length`, so the page said
 * "7 sermões" while the API had been asked for ten.
 *
 * Found when 47 new sermons pushed a golden query's expected result out of a
 * top ten that had three slots taken by duplicates.
 */
const chunk = (sermonId: string, chunkIndex: number, score: number) => ({
  id: `${sermonId}-${chunkIndex}`,
  sermon_id: sermonId,
  chunk_index: chunkIndex,
  content: `trecho ${chunkIndex} de ${sermonId}`,
  score,
  title: `Sermão ${sermonId}`,
  artist: "Reverendo Bruno Melo",
  date: new Date("2025-01-05T00:00:00Z"),
  duration_str: "45:00",
  sc_suffix_url: "slug",
  sp_suffix_url: null,
});

const stubEmbeddings = () => ({ embed: vi.fn(async () => [[0.1, 0.2]]) });

const stubPrisma = (rows: ReturnType<typeof chunk>[]) =>
  ({ $queryRaw: vi.fn(async () => rows) }) as unknown as PrismaClient;

const ids = (results: { id: string }[]) => [...new Set(results.map((r) => r.id))];

describe("search", () => {
  it("fills the page with distinct sermons, not with repeated passages", async () => {
    // Two chunks each from A and B, then C and D. Slicing at 3 used to return
    // A, A, B — two sermons for a three-sermon page.
    const rows = [
      chunk("A", 0, 0.9),
      chunk("A", 4, 0.8),
      chunk("B", 1, 0.7),
      chunk("B", 2, 0.6),
      chunk("C", 0, 0.5),
      chunk("D", 0, 0.4),
    ];

    const { results } = await search(
      { prisma: stubPrisma(rows), embeddings: stubEmbeddings() },
      "q",
      3,
    );

    expect(ids(results)).toEqual(["A", "B", "C"]);
  });

  it("keeps the extra passages of the sermons it does return", async () => {
    // The second passage is the point of max_per_sermon; it just must not cost
    // another sermon its place.
    const rows = [chunk("A", 0, 0.9), chunk("A", 4, 0.8), chunk("B", 1, 0.7)];

    const { results } = await search(
      { prisma: stubPrisma(rows), embeddings: stubEmbeddings() },
      "q",
      2,
    );

    expect(results).toHaveLength(3);
    expect(results.map((r) => r.chunkIndex)).toEqual([0, 4, 1]);
  });

  it("asks the database for enough chunks to fill the page", async () => {
    const prisma = stubPrisma([chunk("A", 0, 0.9)]);
    await search({ prisma, embeddings: stubEmbeddings() }, "q", 10);

    // Asking for ten when a sermon may take two slots cannot fill ten sermons.
    const [call] = (prisma.$queryRaw as unknown as { mock: { calls: unknown[][] } }).mock.calls;
    expect(call?.filter((v) => typeof v === "number")).toContain(20);
  });

  it("preserves the order the ranker chose", async () => {
    const rows = [chunk("C", 0, 0.9), chunk("A", 0, 0.8), chunk("B", 0, 0.7)];

    const { results } = await search(
      { prisma: stubPrisma(rows), embeddings: stubEmbeddings() },
      "q",
      5,
    );

    expect(ids(results)).toEqual(["C", "A", "B"]);
  });

  it("counts sermons the same way after a rerank", async () => {
    const rows = [chunk("A", 0, 0.9), chunk("A", 4, 0.8), chunk("B", 1, 0.7), chunk("C", 0, 0.6)];
    const reranker = {
      rerank: vi.fn(async () => [
        { index: 0, score: 0.99 },
        { index: 1, score: 0.98 },
        { index: 2, score: 0.97 },
        { index: 3, score: 0.96 },
      ]),
    };

    const { results, reranked } = await search(
      { prisma: stubPrisma(rows), embeddings: stubEmbeddings(), reranker },
      "q",
      2,
    );

    expect(reranked).toBe(true);
    expect(ids(results)).toEqual(["A", "B"]);
  });
});
