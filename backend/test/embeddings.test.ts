import { describe, expect, it, vi } from "vitest";
import { createOpenRouterEmbeddings } from "../src/lib/embeddings.ts";

/**
 * The retry/timeout contract. The timeout case is the one that matters most:
 * a full corpus run stalled for good on a dropped connection because fetch
 * had no deadline, so the retry loop never got a chance to run.
 */

const embeddingResponse = (count: number, dims = 1536) =>
  new Response(
    JSON.stringify({
      data: Array.from({ length: count }, (_, index) => ({
        index,
        embedding: Array.from({ length: dims }, () => 0.5),
      })),
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );

describe("createOpenRouterEmbeddings", () => {
  it("returns unit-length vectors", async () => {
    const fetchImpl = vi.fn(async () => embeddingResponse(1));
    const client = createOpenRouterEmbeddings({ apiKey: "k", fetchImpl: fetchImpl as never });

    const [vec] = await client.embed(["texto"]);
    const norm = Math.sqrt((vec ?? []).reduce((acc, x) => acc + x * x, 0));
    expect(norm).toBeCloseTo(1, 6);
  });

  it("aborts a hanging request instead of waiting forever", async () => {
    // Reproduces the stall: fetch that never settles unless aborted.
    const fetchImpl = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
        }),
    );
    const client = createOpenRouterEmbeddings({
      apiKey: "k",
      timeoutMs: 20,
      maxRetries: 1,
      fetchImpl: fetchImpl as never,
    });

    await expect(client.embed(["texto"])).rejects.toThrow(/embedding failed/);
    // Proves it retried rather than hanging on the first attempt.
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("retries on 429 then succeeds", async () => {
    let call = 0;
    const fetchImpl = vi.fn(async () => {
      call++;
      return call === 1 ? new Response("slow down", { status: 429 }) : embeddingResponse(1);
    });
    const client = createOpenRouterEmbeddings({ apiKey: "k", fetchImpl: fetchImpl as never });

    expect(await client.embed(["texto"])).toHaveLength(1);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("does not retry a 4xx request bug", async () => {
    const fetchImpl = vi.fn(async () => new Response("bad model", { status: 400 }));
    const client = createOpenRouterEmbeddings({ apiKey: "k", fetchImpl: fetchImpl as never });

    await expect(client.embed(["texto"])).rejects.toThrow(/OpenRouter 400/);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("rejects when the dimension count is not what was asked for", async () => {
    // Guards the pgvector contract: a 3072-d row cannot go in a halfvec(1536).
    const fetchImpl = vi.fn(async () => embeddingResponse(1, 3072));
    const client = createOpenRouterEmbeddings({ apiKey: "k", fetchImpl: fetchImpl as never });

    await expect(client.embed(["texto"])).rejects.toThrow(/dimensions/);
  });

  it("reorders batched results by their index", async () => {
    const fetchImpl = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            data: [
              { index: 1, embedding: [0, 1] },
              { index: 0, embedding: [1, 0] },
            ],
          }),
          { status: 200 },
        ),
    );
    const client = createOpenRouterEmbeddings({
      apiKey: "k",
      dimensions: 2,
      fetchImpl: fetchImpl as never,
    });

    const [first] = await client.embed(["a", "b"]);
    expect(first?.[0]).toBeCloseTo(1, 6);
  });

  it("short-circuits an empty batch", async () => {
    const fetchImpl = vi.fn();
    const client = createOpenRouterEmbeddings({ apiKey: "k", fetchImpl: fetchImpl as never });

    expect(await client.embed([])).toEqual([]);
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
