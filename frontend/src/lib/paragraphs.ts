/**
 * Break a transcript into readable paragraphs.
 *
 * The corpus has no paragraphs to preserve: `whisperx_worker.py` writes one
 * line per segment, `clean.py` joins them, and the result is a single unbroken
 * block — measured across all 503 committed transcripts, the total number of
 * newline characters is zero. A 6,400-word wall of text is unreadable on a
 * phone, so the breaks are invented here.
 *
 * Invented, and honest about it: the preacher did not pause where a paragraph
 * ends. Sentence boundaries are real (they come from Whisper's punctuation);
 * the grouping into ~100-word blocks is purely typographic. Nothing downstream
 * may treat a paragraph index as a position in the sermon — the chunk offsets
 * that anchor a search result are word-based for exactly that reason.
 */

/**
 * End of sentence: terminal punctuation, optional closing quote/bracket, then
 * whitespace. Kept deliberately loose about what follows — requiring a capital
 * would swallow the break before "e disse o Senhor..." and before every
 * sentence starting with a number, both common in this corpus.
 */
const SENTENCE_END = /(?<=[.!?…]["'”’)\]]?)\s+/;

/** Long enough that paragraphs are not choppy, short enough to scan on a phone. */
const TARGET_WORDS = 100;

function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed === "" ? 0 : trimmed.split(/\s+/).length;
}

export function toParagraphs(text: string, targetWords: number = TARGET_WORDS): string[] {
  const body = text.trim();
  if (body === "") return [];

  const paragraphs: string[] = [];
  let current: string[] = [];
  let words = 0;

  for (const sentence of body.split(SENTENCE_END)) {
    if (sentence.trim() === "") continue;
    current.push(sentence.trim());
    words += countWords(sentence);

    // Close on reaching the target rather than on getting closest to it: a
    // sentence here averages 18 words, so overshoot is small, and the
    // alternative needs lookahead for no visible gain.
    if (words >= targetWords) {
      paragraphs.push(current.join(" "));
      current = [];
      words = 0;
    }
  }

  // A transcript that ends mid-sentence, or one shorter than the target, still
  // has to appear. Dropping the tail is how the last minutes of a sermon would
  // silently stop being readable.
  if (current.length > 0) paragraphs.push(current.join(" "));

  return paragraphs;
}

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
