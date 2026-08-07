-- Browse facets: scripture, series, service type and topics.
--
-- Applied BEFORE 003_hybrid_search.sql, which filters on the columns this
-- file adds to `sermons`. Idempotent -- safe to re-run.
--
-- Hand-written rather than emitted by `prisma migrate diff`, for the reason
-- 000_schema.sql was hand-edited too: the CLI emits unguarded CREATE and ADD
-- CONSTRAINT, and this file is applied verbatim to live data by a
-- postgres:16-alpine sidecar with no Prisma anywhere near it.
--
-- The content of these tables is NOT generated here. It is derived offline,
-- reviewed, and committed under data/facets/; `index-facets.ts` loads it.
-- Production never re-derives, so a parser change cannot alter live data
-- without someone reading the diff first.

-- ---------------------------------------------------------------------------
-- Reference data
-- ---------------------------------------------------------------------------

-- The 66-book canon. `canon_order` is the only sort key the scripture index
-- ever uses: an alphabetical Bible ("Amós, Apocalipse, Atos, Cantares...") is
-- unusable to anyone who knows it.
CREATE TABLE IF NOT EXISTS bible_books (
  slug        text PRIMARY KEY,
  name        text NOT NULL,
  testament   text NOT NULL CHECK (testament IN ('AT', 'NT')),
  canon_order int  NOT NULL,
  chapters    int  NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS bible_books_order_idx ON bible_books (canon_order);

-- Courses and events. `parent_slug` is self-referential so the twelve
-- Westminster chapters hang under the confession that heads them.
CREATE TABLE IF NOT EXISTS series (
  slug         text PRIMARY KEY,
  name         text NOT NULL,
  kind         text NOT NULL,
  parent_slug  text REFERENCES series (slug) ON DELETE SET NULL,
  description  text NOT NULL DEFAULT '',
  sermon_count int  NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS series_parent_idx ON series (parent_slug);

-- Two-level topic taxonomy: the group carries what Desiring God splits into a
-- separate "themes" page, the leaf is the topic itself. One structure is right
-- for 456 sermons; two would be ceremony.
CREATE TABLE IF NOT EXISTS topics (
  slug        text PRIMARY KEY,
  name        text NOT NULL,
  group_slug  text NOT NULL,
  group_name  text NOT NULL,
  description text NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS topics_group_idx ON topics (group_slug);

-- ---------------------------------------------------------------------------
-- Sermon-level facets: single-valued, so they live on the row.
--
-- Being columns rather than a join table is what lets hybrid_search() filter
-- on them without widening its CTEs any further than it must.
-- ---------------------------------------------------------------------------
ALTER TABLE sermons ADD COLUMN IF NOT EXISTS service_type  text;
ALTER TABLE sermons ADD COLUMN IF NOT EXISTS series_slug   text;
ALTER TABLE sermons ADD COLUMN IF NOT EXISTS series_part   int;
ALTER TABLE sermons ADD COLUMN IF NOT EXISTS display_title text;

-- Whether the Spotify episode still resolves. The church's podcast feed is
-- served by SoundCloud and capped at 500 items; every aggregator delists what
-- falls out of that window, so old episodes 404 while their ids stay valid.
-- Derived in data/facets/spotify_episodes.csv. Defaults true so a sermon
-- indexed before the check has run keeps its link rather than losing it.
ALTER TABLE sermons ADD COLUMN IF NOT EXISTS spotify_alive boolean NOT NULL DEFAULT true;

DO $$
BEGIN
  ALTER TABLE sermons
    ADD CONSTRAINT sermons_series_slug_fkey
    FOREIGN KEY (series_slug) REFERENCES series (slug) ON DELETE SET NULL;
EXCEPTION
  WHEN duplicate_object THEN NULL;
  WHEN undefined_column THEN NULL;
END
$$;

CREATE INDEX IF NOT EXISTS sermons_service_type_idx ON sermons (service_type);
CREATE INDEX IF NOT EXISTS sermons_series_idx       ON sermons (series_slug, series_part);

-- ---------------------------------------------------------------------------
-- Multi-valued facets
-- ---------------------------------------------------------------------------

-- One row per (sermon, chapter). A Sunday-school lesson on "Gênesis 12-50"
-- genuinely covers 39 chapters and has to be findable from any of them, so the
-- range is expanded at load time rather than stored as bounds.
--
-- `chapter` is 0, not NULL, when a title names a book without a chapter
-- ("EBD - 1 Samuel"). A sentinel rather than NULL because the row's identity
-- includes the chapter, and Postgres treats NULLs in a unique index as
-- distinct -- two book-level rows for the same sermon would both be allowed.
CREATE TABLE IF NOT EXISTS sermon_scriptures (
  sermon_id   text    NOT NULL REFERENCES sermons (id) ON DELETE CASCADE,
  book_slug   text    NOT NULL REFERENCES bible_books (slug) ON DELETE CASCADE,
  chapter     int     NOT NULL DEFAULT 0,
  verse_start int,
  verse_end   int,
  -- 'titulo' when the title stated it, 'transcricao' when it was recovered
  -- from the sermon audio. Kept so a bad extraction pass can be undone
  -- without touching what the titles established.
  source      text    NOT NULL DEFAULT 'titulo',
  is_primary  boolean NOT NULL DEFAULT true
);

ALTER TABLE sermon_scriptures
  DROP CONSTRAINT IF EXISTS sermon_scriptures_pkey;
ALTER TABLE sermon_scriptures
  ADD PRIMARY KEY (sermon_id, book_slug, chapter);

CREATE INDEX IF NOT EXISTS sermon_scriptures_book_idx    ON sermon_scriptures (book_slug, chapter);
CREATE INDEX IF NOT EXISTS sermon_scriptures_sermon_idx  ON sermon_scriptures (sermon_id);

CREATE TABLE IF NOT EXISTS sermon_topics (
  sermon_id  text  NOT NULL REFERENCES sermons (id) ON DELETE CASCADE,
  topic_slug text  NOT NULL REFERENCES topics (slug) ON DELETE CASCADE,
  confidence float NOT NULL DEFAULT 1.0,
  PRIMARY KEY (sermon_id, topic_slug)
);

CREATE INDEX IF NOT EXISTS sermon_topics_topic_idx ON sermon_topics (topic_slug);
