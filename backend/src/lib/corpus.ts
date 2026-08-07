import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { parseCsv } from "./csv.ts";

/**
 * Reads the sermon corpus produced by the retired GPU pipeline: a metadata CSV
 * plus one cleaned transcript per sermon.
 *
 * The CSV is the output of `sermons_ai/doc_preproc.py`, so the column names and
 * quirks here are inherited, not chosen.
 */

export type SermonRecord = {
  id: string;
  title: string;
  artist: string;
  date: Date;
  durationStr: string;
  durationSec: number;
  scSuffixUrl: string | null;
  spSuffixUrl: string | null;
  score: number;
  words: number;
  sentences: number;
  wordsMin: number;
  sentencesMin: number;
  transcriptFile: string;
};

/** Matches the GPU pipeline's own cutoff, so the corpus is the same one that shipped. */
export const MIN_SCORE = 50;

/**
 * Rejects a transcript that covers only a fraction of its sermon.
 *
 * `score` cannot catch this. It is a mean over the words that *were*
 * transcribed, so a truncated download that transcribes its few minutes
 * cleanly scores as well as a whole sermon -- nine of them reached production
 * that way, scoring 70.8 to 87.1. `words_min` is the orthogonal signal, because
 * `postprocess.py` divides by the true duration from the SoundCloud metadata
 * rather than by the audio that actually downloaded:
 *
 *     words/min   score   audio downloaded
 *          1.4    70.8    2%
 *         16.5    87.1    13%
 *         37.8    86.5    29%
 *         63.1    85.8    47%
 *
 * Approved rows run from 88 (p2) to 139 (median), so 70 clears the slowest real
 * sermon by 18 and catches every confirmed truncation. It is a safety net, not
 * the fix: truncation past ~50% lands above the floor, and only the duration
 * check in `tools/corpus-update/fetch.py` closes that.
 */
export const MIN_WORDS_MIN = 70;

/**
 * Words per minute of sermon, or null when the CSV cannot say.
 *
 * Deliberately fails open on a missing column rather than closed. Reading an
 * absent field as zero would reject every row -- a guard against losing nine
 * sermons that silently loses all five hundred. The column names here are
 * inherited from `doc_preproc.py` and are not ours to rely on absolutely, so
 * `words_min` falls back to the two fields it is derived from.
 */
function wordsPerMinute(r: Record<string, string | undefined>): number | null {
  const stated = toNumber(r.words_min);
  if (stated > 0) return stated;

  const words = toNumber(r.words);
  const minutes = toNumber(r.duration) / 60;
  return words > 0 && minutes > 0 ? words / minutes : null;
}

const isTrue = (v: string | undefined): boolean => (v ?? "").trim().toLowerCase() === "true";

const toNumber = (v: string | undefined): number => {
  const n = Number.parseFloat((v ?? "").trim());
  return Number.isFinite(n) ? n : 0;
};

/**
 * Resolves a sermon's date.
 *
 * Six sermons in the corpus have an empty or corrupt `date` (one reads
 * "0223-05-07" from a typo'd title, five predate the field). Every one of them
 * still has a valid Unix `timestamp`, which is the upstream source the `date`
 * column was derived from -- so fall back to it rather than dropping real
 * sermons over a formatting artifact.
 */
export function resolveDate(
  dateStr: string | undefined,
  timestamp: string | undefined,
): Date | null {
  const raw = (dateStr ?? "").trim();
  const year = Number.parseInt(raw.slice(0, 4), 10);
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw) && year >= 2015 && year <= 2030) {
    return new Date(`${raw}T00:00:00Z`);
  }

  const ts = Number.parseInt((timestamp ?? "").trim(), 10);
  if (Number.isFinite(ts) && ts > 0) {
    const d = new Date(ts * 1000);
    if (d.getUTCFullYear() >= 2015 && d.getUTCFullYear() <= 2030) return d;
  }

  return null;
}

export type LoadResult = {
  sermons: SermonRecord[];
  skipped: { name: string; reason: string }[];
};

/**
 * Loads every sermon that is processed, above the score cutoff, and dated.
 *
 * Deduplicates by transcript file: at least one sermon was uploaded to
 * SoundCloud twice and appears as two rows with different ids pointing at the
 * same transcript. Indexing both would embed the text twice and surface the
 * same sermon twice in every result list. The higher-scoring row wins, ties
 * going to the first seen.
 */
export function loadSermons(csvText: string): LoadResult {
  const rows = parseCsv(csvText);
  const byTranscript = new Map<string, SermonRecord>();
  const sermons: SermonRecord[] = [];
  const skipped: { name: string; reason: string }[] = [];

  for (const r of rows) {
    const name = (r.name ?? "").trim();

    if (!isTrue(r.processed)) {
      skipped.push({ name, reason: "not processed" });
      continue;
    }

    const score = toNumber(r.score);
    if (score <= MIN_SCORE) {
      skipped.push({ name, reason: `score ${score} <= ${MIN_SCORE}` });
      continue;
    }

    const wordsMin = wordsPerMinute(r);
    if (wordsMin !== null && wordsMin <= MIN_WORDS_MIN) {
      skipped.push({ name, reason: `words/min ${wordsMin.toFixed(1)} <= ${MIN_WORDS_MIN}` });
      continue;
    }

    // `txt` already carries the .txt extension -- do not append another.
    const transcriptFile = (r.txt ?? "").trim();
    if (!transcriptFile) {
      skipped.push({ name, reason: "no transcript file" });
      continue;
    }

    const date = resolveDate(r.date, r.timestamp);
    if (!date) {
      skipped.push({ name, reason: "unresolvable date" });
      continue;
    }

    // The SoundCloud id is stable and unique; fall back to the name for the
    // handful of pre-SoundCloud episodes.
    const id = (r.id ?? "").trim() || name;

    const record: SermonRecord = {
      id,
      title: name,
      artist: (r.artist ?? "").trim() || "Desconhecido",
      date,
      durationStr: (r.duration_str ?? "").trim(),
      durationSec: Math.round(toNumber(r.duration)),
      scSuffixUrl: (r.sc_suffix_url ?? "").trim() || null,
      spSuffixUrl: (r.sp_suffix_url ?? "").trim() || null,
      score,
      words: Math.round(toNumber(r.words)),
      sentences: Math.round(toNumber(r.sentences)),
      wordsMin: wordsMin ?? 0,
      sentencesMin: toNumber(r.sentences_min),
      transcriptFile,
    };

    const seen = byTranscript.get(transcriptFile);
    if (seen) {
      // Same sermon, uploaded twice. Keep the better-scoring row.
      if (record.score > seen.score) {
        byTranscript.set(transcriptFile, record);
        sermons[sermons.indexOf(seen)] = record;
        skipped.push({ name: seen.title, reason: `duplicate of ${transcriptFile}` });
      } else {
        skipped.push({ name, reason: `duplicate of ${transcriptFile}` });
      }
      continue;
    }

    byTranscript.set(transcriptFile, record);
    sermons.push(record);
  }

  return { sermons, skipped };
}

export async function readTranscript(dataDir: string, transcriptFile: string): Promise<string> {
  return readFile(join(dataDir, "transcripts", transcriptFile), "utf8");
}
