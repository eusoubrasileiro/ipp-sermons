/**
 * Cross-encoder reranking via OpenRouter.
 *
 * RRF is a strong recall stage but a weak precision stage: it fuses two ranked
 * lists without ever reading the query and passage together. A cross-encoder
 * scores the (query, passage) pair jointly, which is the cheapest large
 * precision win available here. This replaces the local bge-reranker-v2-m3 the
 * GPU pipeline used.
 *
 * Design rule, inherited from the in-house knowledge engine: the rerank call is
 * the seam, and *every* failure mode degrades to the incoming order. A slow,
 * rate-limited or unreachable reranker must never turn a working search into an
 * error page.
 */

export const RERANK_MODEL = "cohere/rerank-4-pro";

const OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank";

/** Reranked positions, referring to indices in the documents array passed in. */
export type RankedIndex = { index: number; score: number };

export type Reranker = {
  /** Returns null on any failure, signalling the caller to keep its own order. */
  rerank(query: string, documents: string[], topN: number): Promise<RankedIndex[] | null>;
};

export type RerankOptions = {
  apiKey: string;
  model?: string;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
  onError?: (err: unknown) => void;
};

export function createOpenRouterReranker(opts: RerankOptions): Reranker {
  const {
    apiKey,
    model = RERANK_MODEL,
    timeoutMs = 5_000,
    fetchImpl = fetch,
    onError = () => {},
  } = opts;

  return {
    async rerank(query, documents, topN) {
      if (documents.length === 0) return [];

      // Bounded wait: search latency is user-facing, and a slow reranker is
      // worse than no reranker.
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);

      try {
        const res = await fetchImpl(OPENROUTER_RERANK_URL, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model,
            query,
            documents,
            top_n: Math.min(topN, documents.length),
          }),
          signal: controller.signal,
        });

        if (!res.ok) {
          onError(new Error(`rerank ${res.status}: ${await res.text()}`));
          return null;
        }

        const body = (await res.json()) as {
          results?: { index: number; relevance_score: number }[];
        };

        if (!body.results?.length) {
          onError(new Error("rerank returned no results"));
          return null;
        }

        // Guard against an out-of-range index poisoning the result list.
        return body.results
          .filter((r) => r.index >= 0 && r.index < documents.length)
          .map((r) => ({ index: r.index, score: r.relevance_score }));
      } catch (err) {
        onError(err);
        return null;
      } finally {
        clearTimeout(timer);
      }
    },
  };
}
