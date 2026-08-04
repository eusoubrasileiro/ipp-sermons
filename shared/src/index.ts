import { z } from "zod";

/**
 * The API contract, owned by `shared/` so backend and frontend cannot drift.
 *
 * Field names mirror the legacy `metadata.csv` columns where they carry the same
 * meaning (`artist` is the preacher, `score` is the wav2vec alignment quality),
 * so the 456 already-transcribed sermons map across without a translation layer.
 */

/** Preacher name, already normalized against `preacher_names.txt` (28 canonical names). */
export const SermonMetaSchema = z.object({
  id: z.string(),
  /** Full sermon title, e.g. "17-09-2021 - A Lei Moral e a Vida Cristã (1)". */
  title: z.string(),
  /** Preacher — legacy `artist` column, e.g. "Reverendo Bruno Melo". */
  artist: z.string(),
  /** ISO date, e.g. "2021-09-17". */
  date: z.string(),
  /** Human-readable runtime, e.g. "1:02:39". */
  durationStr: z.string(),
  /** Playable links — audio is streamed from source, never hosted by us. */
  soundcloudUrl: z.string().url().nullable(),
  spotifyUrl: z.string().url().nullable(),
});
export type SermonMeta = z.infer<typeof SermonMetaSchema>;

/** One retrieved chunk plus the sermon it came from. */
export const SearchResultSchema = SermonMetaSchema.extend({
  /** The matched passage of the transcript. */
  content: z.string(),
  /** Fused relevance score (RRF, or reranker score when reranking is enabled). */
  score: z.number(),
  /** Position of this chunk within its sermon — lets the UI say "trecho 3". */
  chunkIndex: z.number().int().nonnegative(),
});
export type SearchResult = z.infer<typeof SearchResultSchema>;

export const SearchRequestSchema = z.object({
  query: z.string().trim().min(2, "A busca precisa de ao menos 2 caracteres").max(500),
  limit: z.number().int().min(1).max(50).default(10),
});
export type SearchRequest = z.infer<typeof SearchRequestSchema>;

export const SearchResponseSchema = z.object({
  query: z.string(),
  results: z.array(SearchResultSchema),
  /** True when the reranker ran; false when it was disabled or degraded to RRF order. */
  reranked: z.boolean(),
  tookMs: z.number(),
});
export type SearchResponse = z.infer<typeof SearchResponseSchema>;

/** The old server's feedback box, ported from an append-only text file to a table. */
export const SuggestionRequestSchema = z.object({
  suggestion: z.string().trim().min(3).max(2000),
});
export type SuggestionRequest = z.infer<typeof SuggestionRequestSchema>;

export const HealthResponseSchema = z.object({
  status: z.literal("ok"),
  sermons: z.number().int(),
  chunks: z.number().int(),
});
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
