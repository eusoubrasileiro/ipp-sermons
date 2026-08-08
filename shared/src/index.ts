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

/**
 * Browse facets, all optional and all AND-ed together.
 *
 * Filters compose WITH the query rather than replacing it: "briga na igreja"
 * narrowed to Efesios and to one preacher is a single request. That is the
 * point of pushing them into hybrid_search() instead of filtering the results.
 *
 * A filter-only browse with no query does NOT come through here -- `query` has
 * a two-character minimum -- and uses GET /api/sermons instead.
 */
export const SearchFiltersSchema = z.object({
  pregadores: z.array(z.string().min(1)).max(20).optional(),
  tipos: z.array(z.string().min(1)).max(10).optional(),
  series: z.array(z.string().min(1)).max(20).optional(),
  livros: z.array(z.string().min(1)).max(20).optional(),
  capitulo: z.number().int().min(1).max(150).optional(),
  temas: z.array(z.string().min(1)).max(20).optional(),
  de: z
    .string()
    .regex(/^\d{4}-\d{2}-\d{2}$/)
    .optional(),
  ate: z
    .string()
    .regex(/^\d{4}-\d{2}-\d{2}$/)
    .optional(),
});

export type SearchFilters = z.infer<typeof SearchFiltersSchema>;

/** Filter-only browsing: no query text, so it cannot go through the search route. */
export const BrowseRequestSchema = z.object({
  filtros: SearchFiltersSchema.optional(),
  ordenar: z.enum(["data", "biblia", "serie"]).default("data"),
  pagina: z.number().int().min(1).max(500).default(1),
});

export type BrowseRequest = z.infer<typeof BrowseRequestSchema>;

export const SearchRequestSchema = z.object({
  query: z.string().trim().min(2, "A busca precisa de ao menos 2 caracteres").max(500),
  limit: z.number().int().min(1).max(50).default(10),
  filtros: SearchFiltersSchema.optional(),
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

/**
 * A whole sermon, for reading rather than searching.
 *
 * `text` is the transcript exactly as it sits in `data/transcripts/` — one
 * unbroken block, because `clean.py` strips the newlines WhisperX wrote. The
 * paragraphs the reader sees are synthesised in the browser; nothing here
 * pretends the preacher paused where a paragraph breaks.
 *
 * `words` comes along because it is already on the row and lets the page state
 * a reading time without measuring 40 KB of text first.
 */
export const TranscriptResponseSchema = SermonMetaSchema.extend({
  text: z.string(),
  words: z.number().int().nonnegative(),
});
export type TranscriptResponse = z.infer<typeof TranscriptResponseSchema>;

/** The old server's feedback box, ported from an append-only text file to a table. */
export const SuggestionRequestSchema = z.object({
  suggestion: z.string().trim().min(3).max(2000),
});
export type SuggestionRequest = z.infer<typeof SuggestionRequestSchema>;

/**
 * Play-link builders. Shared rather than backend-only because the browse path
 * ships raw suffix columns to the client and rebuilds the URLs there — when
 * this lived in the backend the frontend hand-copied both the SoundCloud
 * channel and the Spotify suppression rule, and the two drifted apart.
 */
export { soundcloudUrl, spotifyUrl } from "./audio-urls.js";

/**
 * Presentation rules shared for the same reason: the server prerenders a
 * sermon's heading, date and paragraphs into `index.html` so a crawler can read
 * them, and the SPA renders them again on hydration. Both sides must write them
 * identically or the page rewrites itself under the reader.
 */
export { formatDate, SITE_TITLE, sermonTitle, stripLeadingDate } from "./format.js";
export { countWords, toParagraphs } from "./paragraphs.js";

export const HealthResponseSchema = z.object({
  status: z.literal("ok"),
  sermons: z.number().int(),
  chunks: z.number().int(),
});
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
