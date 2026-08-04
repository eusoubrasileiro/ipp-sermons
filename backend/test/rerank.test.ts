import { describe, expect, it, vi } from "vitest";
import { createOpenRouterReranker } from "../src/lib/rerank.ts";

/**
 * The contract under test is the fail-safe one: rerank() returns null on every
 * failure path so the caller keeps its existing RRF order. A reranker outage
 * must degrade ranking quality, never break search.
 */

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const docs = ["primeiro", "segundo", "terceiro"];

describe("createOpenRouterReranker", () => {
  it("returns reranked indices on success", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({
        results: [
          { index: 2, relevance_score: 0.9 },
          { index: 0, relevance_score: 0.4 },
        ],
      }),
    );
    const reranker = createOpenRouterReranker({ apiKey: "k", fetchImpl: fetchImpl as never });

    expect(await reranker.rerank("q", docs, 2)).toEqual([
      { index: 2, score: 0.9 },
      { index: 0, score: 0.4 },
    ]);
  });

  it("returns null on a non-ok response", async () => {
    const fetchImpl = vi.fn(async () => new Response("rate limited", { status: 429 }));
    const onError = vi.fn();
    const reranker = createOpenRouterReranker({
      apiKey: "k",
      fetchImpl: fetchImpl as never,
      onError,
    });

    expect(await reranker.rerank("q", docs, 3)).toBeNull();
    expect(onError).toHaveBeenCalled();
  });

  it("returns null when the network throws", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error("ECONNREFUSED");
    });
    const reranker = createOpenRouterReranker({ apiKey: "k", fetchImpl: fetchImpl as never });

    expect(await reranker.rerank("q", docs, 3)).toBeNull();
  });

  it("returns null when the body has no results", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ results: [] }));
    const reranker = createOpenRouterReranker({ apiKey: "k", fetchImpl: fetchImpl as never });

    expect(await reranker.rerank("q", docs, 3)).toBeNull();
  });

  it("drops out-of-range indices rather than returning a bad position", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({
        results: [
          { index: 99, relevance_score: 0.9 },
          { index: 1, relevance_score: 0.5 },
        ],
      }),
    );
    const reranker = createOpenRouterReranker({ apiKey: "k", fetchImpl: fetchImpl as never });

    expect(await reranker.rerank("q", docs, 2)).toEqual([{ index: 1, score: 0.5 }]);
  });

  it("short-circuits on an empty document list", async () => {
    const fetchImpl = vi.fn();
    const reranker = createOpenRouterReranker({ apiKey: "k", fetchImpl: fetchImpl as never });

    expect(await reranker.rerank("q", [], 5)).toEqual([]);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("caps top_n at the number of documents", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({ results: [{ index: 0, relevance_score: 1 }] }),
    );
    const reranker = createOpenRouterReranker({ apiKey: "k", fetchImpl: fetchImpl as never });

    await reranker.rerank("q", docs, 100);
    const init = fetchImpl.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(init).toBeDefined();
    const body = JSON.parse(String(init?.body));
    expect(body.top_n).toBe(3);
  });
});
