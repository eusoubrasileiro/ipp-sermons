-- Portuguese hybrid search: lexical (tsvector) + semantic (pgvector), fused by RRF.
--
-- Applied after `prisma db push`. Idempotent — safe to re-run.
--
-- Why hybrid and not pure vector: this corpus is full of proper nouns and
-- scripture references ("Eclesiastes 3.2b-3a", "Jonathan Edwards", a preacher's
-- name). Embeddings blur exact tokens; BM25-style lexical matching nails them.
-- The GPU system this replaces used the same two-arm design, and the in-house
-- knowledge-engine independently reached the same conclusion.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- ---------------------------------------------------------------------------
-- Portuguese FTS config with accent folding.
--
-- Sermon transcripts are accented Portuguese, but people search without accents
-- ("justificacao pela fe"). unaccent() folds both sides so they meet.
--
-- This must be a *named configuration* rather than a bare unaccent() call:
-- unaccent() is STABLE, not IMMUTABLE, so Postgres rejects it inside a GENERATED
-- column. to_tsvector(regconfig, text) with a fixed config IS immutable, which
-- is what makes the stored column below legal.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = 'pt_unaccent') THEN
    CREATE TEXT SEARCH CONFIGURATION pt_unaccent (COPY = portuguese);
    ALTER TEXT SEARCH CONFIGURATION pt_unaccent
      ALTER MAPPING FOR hword, hword_part, word
      WITH unaccent, portuguese_stem;
  END IF;
END
$$;

-- Generated tsvector, maintained by Postgres on every write.
ALTER TABLE sermon_chunks
  ADD COLUMN IF NOT EXISTS fts tsvector
  GENERATED ALWAYS AS (to_tsvector('pt_unaccent', content)) STORED;

CREATE INDEX IF NOT EXISTS sermon_chunks_fts_idx
  ON sermon_chunks USING gin (fts);

-- HNSW over cosine distance. ~20,400 chunks x 1536 halfvec is ~126 MB resident,
-- which matters: the VPS has ~2 GB free.
CREATE INDEX IF NOT EXISTS sermon_chunks_embedding_idx
  ON sermon_chunks USING hnsw (embedding halfvec_cosine_ops);

-- ---------------------------------------------------------------------------
-- hybrid_search — two ranked lists fused by Reciprocal Rank Fusion.
--
-- RRF scores by *rank position*, not raw score, which is why it can merge two
-- incomparable scales (ts_rank_cd relevance vs cosine distance) with no tuning
-- or normalization. Each arm contributes 1/(k + rank).
--
-- k = 60 is the value from the original RRF paper and the ParadeDB/Supabase
-- recipes: large enough that the top few ranks don't dominate outright.
--
-- Both arms over-fetch (match_count * 2) before fusion. Fusing wide then
-- trimming beats fusing exactly `match_count` from each side, because a
-- document ranked 15th lexically and 15th semantically should still surface.
-- ---------------------------------------------------------------------------
-- Filtering note (the whole reason the predicate appears three times):
-- each arm truncates at match_count * 8 (or * 2) BEFORE the fusion below.
-- Filtering after the fusion, or in the caller's outer SELECT, would throw
-- away most of an already-truncated candidate pool and return two or three
-- results for a perfectly ordinary filtered query -- with no error anywhere.
-- The predicate has to narrow each pool as it is built, not after.
--
-- Query-building note (learned the hard way, keep this):
-- websearch_to_tsquery ANDs bare terms, so "briga na igreja" becomes
-- 'brig' & 'igrej' and matches only chunks containing BOTH stems. On a sermon
-- corpus that silently zeroes the lexical arm for most multi-word queries,
-- leaving "hybrid" search running on the vector arm alone. ipp_to_tsquery
-- rewrites the top-level ANDs into ORs so partial matches still rank, and
-- ts_rank_cd then rewards chunks that hit more of the terms.
-- Quoted phrases keep their <-> (phrase) operators untouched.
CREATE OR REPLACE FUNCTION ipp_to_tsquery(query_text text)
RETURNS tsquery
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT to_tsquery(
    'pt_unaccent',
    replace(websearch_to_tsquery('pt_unaccent', query_text)::text, ' & ', ' | ')
  );
$$;

-- The pre-facet signature has to go before the new one is created: adding
-- parameters with defaults makes an OVERLOAD, not a replacement, and a call
-- with the old argument count would then be ambiguous and fail at runtime.
DROP FUNCTION IF EXISTS hybrid_search(text, halfvec, int, float, float, int, int, float);

CREATE OR REPLACE FUNCTION hybrid_search(
  query_text      text,
  query_embedding halfvec(1536),
  match_count     int   DEFAULT 10,
  full_text_weight  float DEFAULT 1.0,
  semantic_weight   float DEFAULT 1.0,
  rrf_k           int   DEFAULT 60,
  -- Most chunks any one sermon may contribute to the result list. 2 keeps a
  -- second supporting passage while leaving room for other sermons.
  max_per_sermon  int   DEFAULT 2,
  -- Titles are curated metadata (passage + topic), not transcribed speech, so
  -- a title hit is a strong signal and is weighted above the two text arms.
  title_weight    float DEFAULT 1.5,
  -- Browse facets. NULL means "no constraint", so every existing call is
  -- unaffected. Bound parameters, never interpolated -- unlike the vector,
  -- whose type modifier Postgres requires to be a literal.
  filter_artists  text[] DEFAULT NULL,
  filter_types    text[] DEFAULT NULL,
  filter_series   text[] DEFAULT NULL,
  filter_books    text[] DEFAULT NULL,
  filter_chapter  int    DEFAULT NULL,
  filter_topics   text[] DEFAULT NULL,
  filter_from     date   DEFAULT NULL,
  filter_to       date   DEFAULT NULL
)
RETURNS TABLE (
  id            text,
  sermon_id     text,
  chunk_index   int,
  content       text,
  score         float
)
LANGUAGE sql
STABLE
AS $$
  -- Sermons whose *title* matches.
  --
  -- A preacher rarely says the title out loud, so the transcript of "O sétimo
  -- mandamento" never contains that phrase and the sermon ranked 52nd on chunk
  -- text alone. Titles carry the passage and topic a person actually searches
  -- for, so match them separately and let the sermon's best chunk represent it.
  WITH title_match AS (
    SELECT
      (SELECT sc.id
         FROM sermon_chunks sc
        WHERE sc.sermon_id = s.id
        ORDER BY ts_rank_cd(sc.fts, ipp_to_tsquery(query_text)) DESC, sc.chunk_index
        LIMIT 1) AS id,
      row_number() OVER (
        ORDER BY ts_rank_cd(to_tsvector('pt_unaccent', s.title), ipp_to_tsquery(query_text)) DESC
      ) AS rank_ix
    FROM sermons s
    WHERE to_tsvector('pt_unaccent', s.title) @@ ipp_to_tsquery(query_text)
      AND (filter_artists IS NULL OR s.artist      = ANY(filter_artists))
      AND (filter_types   IS NULL OR s.service_type = ANY(filter_types))
      AND (filter_series  IS NULL OR s.series_slug  = ANY(filter_series))
      AND (filter_from    IS NULL OR s.date        >= filter_from)
      AND (filter_to      IS NULL OR s.date        <= filter_to)
      AND (filter_books   IS NULL OR EXISTS (
             SELECT 1 FROM sermon_scriptures ss
              WHERE ss.sermon_id = s.id
                AND ss.book_slug = ANY(filter_books)
                AND (filter_chapter IS NULL OR ss.chapter = filter_chapter)))
      AND (filter_topics  IS NULL OR EXISTS (
             SELECT 1 FROM sermon_topics st
              WHERE st.sermon_id = s.id
                AND st.topic_slug = ANY(filter_topics)))
    ORDER BY rank_ix
    LIMIT match_count * 2
  ),
  full_text AS (
    SELECT
      sc.id,
      row_number() OVER (
        ORDER BY ts_rank_cd(sc.fts, ipp_to_tsquery(query_text)) DESC
      ) AS rank_ix
    FROM sermon_chunks sc
    JOIN sermons s ON s.id = sc.sermon_id
    WHERE sc.fts @@ ipp_to_tsquery(query_text)
      AND (filter_artists IS NULL OR s.artist      = ANY(filter_artists))
      AND (filter_types   IS NULL OR s.service_type = ANY(filter_types))
      AND (filter_series  IS NULL OR s.series_slug  = ANY(filter_series))
      AND (filter_from    IS NULL OR s.date        >= filter_from)
      AND (filter_to      IS NULL OR s.date        <= filter_to)
      AND (filter_books   IS NULL OR EXISTS (
             SELECT 1 FROM sermon_scriptures ss
              WHERE ss.sermon_id = s.id
                AND ss.book_slug = ANY(filter_books)
                AND (filter_chapter IS NULL OR ss.chapter = filter_chapter)))
      AND (filter_topics  IS NULL OR EXISTS (
             SELECT 1 FROM sermon_topics st
              WHERE st.sermon_id = s.id
                AND st.topic_slug = ANY(filter_topics)))
    ORDER BY rank_ix
    LIMIT match_count * 8
  ),
  semantic AS (
    SELECT
      sc.id,
      row_number() OVER (ORDER BY sc.embedding <=> query_embedding) AS rank_ix
    FROM sermon_chunks sc
    JOIN sermons s ON s.id = sc.sermon_id
    WHERE sc.embedding IS NOT NULL
      AND (filter_artists IS NULL OR s.artist      = ANY(filter_artists))
      AND (filter_types   IS NULL OR s.service_type = ANY(filter_types))
      AND (filter_series  IS NULL OR s.series_slug  = ANY(filter_series))
      AND (filter_from    IS NULL OR s.date        >= filter_from)
      AND (filter_to      IS NULL OR s.date        <= filter_to)
      AND (filter_books   IS NULL OR EXISTS (
             SELECT 1 FROM sermon_scriptures ss
              WHERE ss.sermon_id = s.id
                AND ss.book_slug = ANY(filter_books)
                AND (filter_chapter IS NULL OR ss.chapter = filter_chapter)))
      AND (filter_topics  IS NULL OR EXISTS (
             SELECT 1 FROM sermon_topics st
              WHERE st.sermon_id = s.id
                AND st.topic_slug = ANY(filter_topics)))
    ORDER BY rank_ix
    LIMIT match_count * 8
  ),
  -- FULL OUTER JOIN is load-bearing: a chunk found by only one arm must still
  -- compete. An INNER JOIN would silently discard every single-arm hit, which
  -- is exactly the recall the hybrid design exists to capture.
  fused AS (
    SELECT
      sc.id,
      sc.sermon_id,
      sc.chunk_index,
      sc.content,
      (
        COALESCE(1.0 / (rrf_k + full_text.rank_ix),   0.0) * full_text_weight +
        COALESCE(1.0 / (rrf_k + semantic.rank_ix),    0.0) * semantic_weight +
        COALESCE(1.0 / (rrf_k + title_match.rank_ix), 0.0) * title_weight
      )::float AS score
    FROM full_text
    FULL OUTER JOIN semantic    ON full_text.id = semantic.id
    FULL OUTER JOIN title_match ON title_match.id = COALESCE(full_text.id, semantic.id)
    JOIN sermon_chunks sc
      ON sc.id = COALESCE(full_text.id, semantic.id, title_match.id)
  ),
  -- Diversify by sermon.
  --
  -- Ranking raw chunks means a long sermon can take most of the result list:
  -- one 23-chunk sermon filled the top 10 for "Eclesiastes tempo de plantar"
  -- and buried the other half of the same series. People search for sermons,
  -- not passages, so cap each sermon's share and let more of them surface.
  -- The candidate pools above are widened to compensate for what this drops.
  ranked AS (
    SELECT
      f.*,
      row_number() OVER (PARTITION BY f.sermon_id ORDER BY f.score DESC) AS per_sermon
    FROM fused f
  )
  SELECT r.id, r.sermon_id, r.chunk_index, r.content, r.score
  FROM ranked r
  WHERE r.per_sermon <= max_per_sermon
  ORDER BY r.score DESC
  LIMIT match_count;
$$;
