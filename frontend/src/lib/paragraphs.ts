import { countWords } from "@ipp/shared";

/**
 * Where a transcript breaks, and where a search result sits inside it.
 *
 * The splitting itself lives in `@ipp/shared`: the server renders the same
 * paragraphs into the page a crawler reads, and React renders them again on
 * hydration, so a second copy of the rule would reflow the article under the
 * reader. Anchoring is the browser's business alone and stays here.
 */
export { toParagraphs } from "@ipp/shared";

/**
 * Which paragraph holds the given word of the transcript.
 *
 * The search result carries `chunkIndex`, and `chunkText(text, 200, 30)` steps
 * 170 words at a time, so chunk `i` opens at word `i * 170`. Arithmetic rather
 * than searching for the chunk's text: the chunk was built from whitespace-
 * normalised words and would not always match the file byte for byte.
 *
 * Returns 0 when the offset runs past the end, which is the honest answer for a
 * stale link — the top of the sermon, not a blank screen.
 */
export function paragraphAtWord(paragraphs: string[], wordOffset: number): number {
  if (wordOffset <= 0) return 0;
  let seen = 0;
  for (const [index, paragraph] of paragraphs.entries()) {
    seen += countWords(paragraph);
    if (seen > wordOffset) return index;
  }
  return 0;
}

/** Words per chunk step in `chunkText(text, 200, 30)`. */
export const CHUNK_STRIDE = 170;
