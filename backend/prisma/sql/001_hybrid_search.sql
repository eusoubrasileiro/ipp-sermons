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

CREATE OR REPLACE FUNCTION hybrid_search(
  query_text      text,
  query_embedding halfvec(1536),
  match_count     int   DEFAULT 10,
  full_text_weight  float DEFAULT 1.0,
  semantic_weight   float DEFAULT 1.0,
  rrf_k           int   DEFAULT 60
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
  WITH full_text AS (
    SELECT
      sc.id,
      row_number() OVER (
        ORDER BY ts_rank_cd(sc.fts, ipp_to_tsquery(query_text)) DESC
      ) AS rank_ix
    FROM sermon_chunks sc
    WHERE sc.fts @@ ipp_to_tsquery(query_text)
    ORDER BY rank_ix
    LIMIT match_count * 2
  ),
  semantic AS (
    SELECT
      sc.id,
      row_number() OVER (ORDER BY sc.embedding <=> query_embedding) AS rank_ix
    FROM sermon_chunks sc
    WHERE sc.embedding IS NOT NULL
    ORDER BY rank_ix
    LIMIT match_count * 2
  )
  -- FULL OUTER JOIN is load-bearing: a chunk found by only one arm must still
  -- compete. An INNER JOIN would silently discard every single-arm hit, which
  -- is exactly the recall the hybrid design exists to capture.
  SELECT
    sc.id,
    sc.sermon_id,
    sc.chunk_index,
    sc.content,
    (
      COALESCE(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
      COALESCE(1.0 / (rrf_k + semantic.rank_ix),  0.0) * semantic_weight
    )::float AS score
  FROM full_text
  FULL OUTER JOIN semantic ON full_text.id = semantic.id
  JOIN sermon_chunks sc ON sc.id = COALESCE(full_text.id, semantic.id)
  ORDER BY score DESC
  LIMIT match_count;
$$;
