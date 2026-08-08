import { basename } from "node:path";
import { soundcloudUrl, spotifyUrl, type TranscriptResponse } from "@ipp/shared";
import type { PrismaClient } from "@prisma/client";
import { readTranscript } from "./corpus.ts";

/**
 * A whole sermon, for reading rather than searching.
 *
 * The text is NOT in Postgres. `data/transcripts/` already ships inside the
 * image (see the Dockerfile), so a second copy in the database would cost 20 MB
 * to buy nothing, and the row carries only the file name — which cannot be
 * derived from the title, since `txt` equals `name + ".txt"` in just 456 of the
 * 514 corpus rows.
 *
 * Everything that can go wrong returns null rather than throwing, because none
 * of it is exceptional: a row indexed before the column existed has no file
 * name until the next index run, and a corpus update can remove a transcript a
 * row still points at. Both are a 404 to the reader, not a 500.
 */

/** Rebuilt at read time; nothing stores a full URL. */
function playLinks(row: {
  scSuffixUrl: string | null;
  spSuffixUrl: string | null;
  spotifyAlive: boolean;
}) {
  return {
    soundcloudUrl: soundcloudUrl(row.scSuffixUrl),
    spotifyUrl: spotifyUrl(row.spSuffixUrl, row.spotifyAlive),
  };
}

export async function readSermonTranscript(
  prisma: PrismaClient,
  dataDir: string,
  id: string,
): Promise<TranscriptResponse | null> {
  const row = await prisma.sermon.findUnique({
    where: { id },
    select: {
      id: true,
      title: true,
      artist: true,
      date: true,
      durationStr: true,
      scSuffixUrl: true,
      spSuffixUrl: true,
      spotifyAlive: true,
      words: true,
      transcriptFile: true,
    },
  });
  if (!row?.transcriptFile) return null;

  // The name comes from the database, not the URL, so this is defence in depth
  // -- but it lands in a path join, and the corpus is rebuilt from a CSV that no
  // schema validates. Anything that is not a plain file name in that directory
  // is treated as absent.
  if (basename(row.transcriptFile) !== row.transcriptFile) return null;

  let text: string;
  try {
    text = await readTranscript(dataDir, row.transcriptFile);
  } catch {
    return null;
  }

  return {
    id: row.id,
    title: row.title,
    artist: row.artist,
    date: row.date.toISOString().slice(0, 10),
    durationStr: row.durationStr,
    ...playLinks(row),
    words: row.words,
    text,
  };
}
