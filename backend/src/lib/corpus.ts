import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

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
 * Minimal RFC-4180 CSV parser.
 *
 * Hand-rolled rather than pulled from npm because sermon descriptions contain
 * embedded commas, quotes and newlines, and this is the one file we ever parse.
 */
export function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];

    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (ch !== "\r") {
      field += ch;
    }
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  const header = rows.shift();
  if (!header) return [];

  return rows
    .filter((r) => r.some((c) => c.trim() !== ""))
    .map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""])));
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
      wordsMin: toNumber(r.words_min),
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

/**
 * Splits a transcript into overlapping word windows.
 *
 * 200 words with 30 of overlap reproduces the chunking the GPU pipeline used
 * (`sermons_ai/doc_embedder.py`), so retrieval behaviour stays comparable to
 * the system this replaces. The overlap keeps a thought that straddles a
 * boundary retrievable from either side.
 */
export function chunkText(text: string, chunkWords = 200, overlapWords = 30): string[] {
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  if (words.length === 0) return [];
  if (words.length <= chunkWords) return [words.join(" ")];

  const stride = chunkWords - overlapWords;
  const chunks: string[] = [];

  for (let start = 0; start < words.length; start += stride) {
    const slice = words.slice(start, start + chunkWords);
    // Drop a trailing sliver that the previous chunk's overlap already covers.
    if (slice.length < overlapWords && chunks.length > 0) break;
    chunks.push(slice.join(" "));
    if (start + chunkWords >= words.length) break;
  }

  return chunks;
}

/** Identity of a chunk's content, so re-indexing can skip unchanged work. */
export function chunkHash(sermonId: string, chunkIndex: number, content: string): string {
  return createHash("sha256").update(`${sermonId}:${chunkIndex}:${content}`).digest("hex");
}
