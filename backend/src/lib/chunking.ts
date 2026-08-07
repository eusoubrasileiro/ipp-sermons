import { createHash } from "node:crypto";

/**
 * Splitting a transcript into the windows that get embedded.
 *
 * Separate from `corpus.ts`, which reads what the Python pipeline produced.
 * This is the indexer's concern -- only `scripts/index-corpus.ts` uses it --
 * and its parameters answer to the embedding model, not to the CSV.
 */

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
