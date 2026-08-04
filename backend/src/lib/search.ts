import type { SearchResult } from "@ipp/shared";
import pkg, { type PrismaClient } from "@prisma/client";

// @prisma/client is CommonJS: `import { Prisma }` resolves under the dev
// loader but not in the compiled ESM build, where named-export detection
// misses it. Destructure from the default export instead.
const { Prisma } = pkg;

import { EMBEDDING_DIMS, type EmbeddingsClient, toVectorLiteral } from "./embeddings.ts";
import type { Reranker } from "./rerank.ts";

/**
 * Hybrid retrieval: lexical + semantic, fused in SQL, then optionally reranked.
 *
 * The fusion itself lives in the `hybrid_search()` Postgres function rather
 * than in application code -- Postgres is the only place that can rank both
 * arms without shipping the whole corpus over the wire.
 */

const SOUNDCLOUD_BASE = "https://soundcloud.com";
const SPOTIFY_BASE = "https://open.spotify.com/episode";

type HybridRow = {
  id: string;
  sermon_id: string;
  chunk_index: number;
  content: string;
  score: number;
  title: string;
  artist: string;
  date: Date;
  duration_str: string;
  sc_suffix_url: string | null;
  sp_suffix_url: string | null;
};

/**
 * How many candidates to pull before reranking.
 *
 * The reranker is a far better judge of relevance than RRF, but it can only
 * reorder what it is given -- so feed it a wide net and let it pick. Without a
 * reranker this is trimmed straight back to `limit`.
 */
const RERANK_CANDIDATES = 40;

export type SearchDeps = {
  prisma: PrismaClient;
  embeddings: EmbeddingsClient;
  reranker?: Reranker | undefined;
};

export type SearchOutcome = {
  results: SearchResult[];
  reranked: boolean;
};

function toSearchResult(row: HybridRow, score: number): SearchResult {
  return {
    id: row.sermon_id,
    title: row.title,
    artist: row.artist,
    date: row.date.toISOString().slice(0, 10),
    durationStr: row.duration_str,
    soundcloudUrl: row.sc_suffix_url ? `${SOUNDCLOUD_BASE}/${row.sc_suffix_url}` : null,
    spotifyUrl: row.sp_suffix_url ? `${SPOTIFY_BASE}/${row.sp_suffix_url}` : null,
    content: row.content,
    score,
    chunkIndex: row.chunk_index,
  };
}

export async function search(
  deps: SearchDeps,
  query: string,
  limit: number,
): Promise<SearchOutcome> {
  const [queryVector] = await deps.embeddings.embed([query]);
  if (!queryVector) throw new Error("failed to embed query");

  const candidateCount = deps.reranker ? RERANK_CANDIDATES : limit;

  // The vector literal and dimension are interpolated (the dimension because
  // Postgres requires literal type modifiers); both are our own values, never
  // user input. `query` and `candidateCount` stay bound.
  const rows = await deps.prisma.$queryRaw<HybridRow[]>`
    SELECT h.id, h.sermon_id, h.chunk_index, h.content, h.score,
           s.title, s.artist, s.date, s.duration_str, s.sc_suffix_url, s.sp_suffix_url
    FROM hybrid_search(
           ${query},
           ${Prisma.raw(`'${toVectorLiteral(queryVector)}'::halfvec(${EMBEDDING_DIMS})`)},
           ${candidateCount}::int
         ) h
    JOIN sermons s ON s.id = h.sermon_id
    ORDER BY h.score DESC`;

  if (rows.length === 0) return { results: [], reranked: false };

  if (!deps.reranker) {
    return {
      results: rows.slice(0, limit).map((r) => toSearchResult(r, r.score)),
      reranked: false,
    };
  }

  // A reranker failure must never fail the search: fall back to the RRF order
  // we already have. Degrading to a slightly worse ranking beats an error page.
  const ranked = await deps.reranker.rerank(
    query,
    rows.map((r) => r.content),
    limit,
  );

  if (!ranked) {
    return {
      results: rows.slice(0, limit).map((r) => toSearchResult(r, r.score)),
      reranked: false,
    };
  }

  const results: SearchResult[] = [];
  for (const { index, score } of ranked) {
    const row = rows[index];
    if (row) results.push(toSearchResult(row, score));
  }

  return { results, reranked: true };
}
