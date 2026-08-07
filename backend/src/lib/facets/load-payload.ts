import { assertMatched, int } from "./csv.ts";

/**
 * The rows a facet load is about to write, computed before anything is deleted.
 *
 * `sermon_scriptures` and `sermon_topics` are replaced wholesale rather than
 * merged, because the derivation is authoritative: a chapter that stopped being
 * derived has to stop being listed. That makes the ordering load-bearing.
 * These functions filter and validate with no database access, so the caller
 * can hold a complete, checked payload in hand before it deletes anything —
 * and can do both inside one transaction.
 *
 * The alternative, deleting first and discovering the payload was empty
 * afterwards, is not hypothetical: it is what the loader used to do, and the
 * failure lands in the one-shot `facets` service on a production deploy, where
 * the site keeps rendering and passing its health check with nothing in it.
 */
type ScripturePayload = {
  sermonId: string;
  bookSlug: string;
  chapter: number;
  verseStart: number | null;
  verseEnd: number | null;
  source: string;
  isPrimary: boolean;
};

type TopicPayload = {
  sermonId: string;
  topicSlug: string;
  confidence: number;
};

const trim = (v: string | undefined): string => (v ?? "").trim();

/**
 * @param sermons ids already in the database — index-corpus has to run first,
 *   or a new sermon's passages are silently dropped here.
 */
export function scripturePayload(
  rows: Record<string, string>[],
  sermons: Set<string>,
  books: Set<string>,
): ScripturePayload[] {
  const payload = rows
    .filter((r) => sermons.has(trim(r.sermon_id)) && books.has(trim(r.book_slug)))
    .map((r) => ({
      sermonId: trim(r.sermon_id),
      bookSlug: trim(r.book_slug),
      chapter: int(r.chapter) ?? 0,
      verseStart: int(r.verse_start),
      verseEnd: int(r.verse_end),
      source: trim(r.source) || "titulo",
      isPrimary: trim(r.is_primary) !== "false",
    }));

  assertMatched("sermon_scriptures.csv (sermon_id, book_slug)", rows.length, payload.length);
  return payload;
}

export function topicPayload(
  rows: Record<string, string>[],
  sermons: Set<string>,
  topics: Set<string>,
): TopicPayload[] {
  const payload = rows
    .filter((r) => sermons.has(trim(r.sermon_id)) && topics.has(trim(r.topico_slug)))
    .map((r) => ({
      sermonId: trim(r.sermon_id),
      topicSlug: trim(r.topico_slug),
      confidence: Number.parseFloat(r.confianca ?? "1") || 1,
    }));

  assertMatched("sermon_topics.csv (sermon_id, topico_slug)", rows.length, payload.length);
  return payload;
}

/**
 * Splits `spotify_episodes.csv` into the sermon ids whose episode still
 * resolves and those whose episode has aged out of the 500-item podcast feed.
 *
 * A partition rather than a per-row update: the file covers every sermon that
 * has an episode id at all, so two `updateMany`s settle it.
 */
export function spotifyPartition(rows: Record<string, string>[]): {
  alive: string[];
  dead: string[];
} {
  const ids = (alive: boolean) =>
    rows.filter((r) => (trim(r.alive) === "true") === alive).map((r) => trim(r.sermon_id));
  return { alive: ids(true), dead: ids(false) };
}
