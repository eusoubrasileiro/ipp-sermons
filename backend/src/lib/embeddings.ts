/**
 * Embeddings via OpenRouter.
 *
 * OpenRouter is the single LLM credential for every project here, so this goes
 * through it rather than talking to Google directly.
 *
 * Model choice: gemini-embedding-001 ranks first among commercial APIs on
 * MTEB-BR, the native Brazilian-Portuguese benchmark (arXiv 2607.04581).
 * Global multilingual leaderboards are a poor proxy for pt-BR -- they correlate
 * only moderately, so the usual default (text-embedding-3-*) is not the right
 * pick for a Portuguese corpus.
 */

import { FatalOpenRouterError, postJson, RetryableError } from "./openrouter.ts";
import type { Usage } from "./usage.ts";

export const EMBEDDING_MODEL = "google/gemini-embedding-001";

/**
 * Native output is 3072-d, truncated here via Matryoshka.
 *
 * 1536 is a hard requirement, not a preference: pgvector caps HNSW indexes on
 * `vector` at 2000 dimensions, so 3072 could not be indexed at all. Measured
 * cost of the truncation on Portuguese pairs is ~0.01-0.02 cosine -- negligible.
 */
export const EMBEDDING_DIMS = 1536;

const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";

export type EmbeddingsClient = {
  embed(inputs: string[]): Promise<number[][]>;
};

/**
 * Rescales a vector to unit length.
 *
 * Load-bearing. Truncated Matryoshka vectors come back with an L2 norm around
 * 0.697, NOT 1.0 -- truncation drops magnitude along with the tail dimensions.
 * pgvector's cosine operator assumes unit vectors, so storing them raw would
 * skew every similarity in the index. Nothing would error; results would just
 * be quietly wrong.
 */
export function normalize(vec: number[]): number[] {
  let sumSquares = 0;
  for (const x of vec) sumSquares += x * x;
  const norm = Math.sqrt(sumSquares);
  if (norm === 0) return vec;
  return vec.map((x) => x / norm);
}

/**
 * An error that retrying cannot fix: a malformed request, or a response whose
 * shape violates the pgvector contract. Every attempt costs money, so these
 * bypass the retry loop entirely.
 */
export class FatalEmbeddingError extends FatalOpenRouterError {
  override readonly name = "FatalEmbeddingError";
}

export type OpenRouterOptions = {
  apiKey: string;
  model?: string;
  dimensions?: number;
  maxRetries?: number;
  /**
   * Per-attempt ceiling. Without this a dropped connection leaves fetch
   * pending forever: the indexer parks in ep_poll holding no socket, makes no
   * progress, and never errors -- so the retry loop never even runs. Observed
   * for real partway through a full corpus run.
   */
  timeoutMs?: number;
  retryDelayMs?: number;
  fetchImpl?: typeof fetch;
  /** Where a run adds up what it spent — see lib/usage.ts. */
  onUsage?: (usage: Usage | null) => void;
};

export function createOpenRouterEmbeddings(opts: OpenRouterOptions): EmbeddingsClient {
  const {
    apiKey,
    model = EMBEDDING_MODEL,
    dimensions = EMBEDDING_DIMS,
    onUsage,
    maxRetries = 5,
    timeoutMs = 60_000,
    retryDelayMs,
    fetchImpl,
  } = opts;

  return {
    async embed(inputs: string[]): Promise<number[][]> {
      if (inputs.length === 0) return [];

      return postJson(
        {
          url: OPENROUTER_URL,
          apiKey,
          body: { model, input: inputs, dimensions, usage: { include: true } },
          label: "embedding",
          maxRetries,
          timeoutMs,
          ...(retryDelayMs === undefined ? {} : { retryDelayMs }),
          ...(fetchImpl === undefined ? {} : { fetchImpl }),
          ...(onUsage === undefined ? {} : { onUsage }),
        },
        (payload) => {
          const body = payload as {
            data?: { embedding: number[]; index?: number }[];
            error?: unknown;
          };

          if (body.error)
            throw new RetryableError(`OpenRouter error: ${JSON.stringify(body.error)}`);
          if (!body.data || body.data.length !== inputs.length) {
            throw new RetryableError(
              `expected ${inputs.length} embeddings, got ${body.data?.length ?? 0}`,
            );
          }

          // The API may return rows out of order; `index` is authoritative.
          const ordered = [...body.data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));

          return ordered.map((row) => {
            if (row.embedding.length !== dimensions) {
              throw new FatalEmbeddingError(
                `expected ${dimensions} dimensions, got ${row.embedding.length} -- ` +
                  "the `dimensions` parameter was not honoured",
              );
            }
            return normalize(row.embedding);
          });
        },
      );
    },
  };
}

/** pgvector literal form: `[0.1,0.2,...]`. */
export function toVectorLiteral(vec: number[]): string {
  return `[${vec.map((x) => x.toFixed(6)).join(",")}]`;
}
