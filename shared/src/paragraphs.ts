/**
 * Break a transcript into readable paragraphs.
 *
 * The corpus has no paragraphs to preserve: `whisperx_worker.py` writes one
 * line per segment, `clean.py` joins them, and the result is a single unbroken
 * block — measured across all 503 committed transcripts, the total number of
 * newline characters is zero. A 6,400-word wall of text is unreadable on a
 * phone, and a single 40 KB `<p>` is no better to a search engine, so the
 * breaks are invented here.
 *
 * Invented, and honest about it: the preacher did not pause where a paragraph
 * ends. Sentence boundaries are real (they come from Whisper's punctuation);
 * the grouping into ~100-word blocks is purely typographic. Nothing downstream
 * may treat a paragraph index as a position in the sermon — the chunk offsets
 * that anchor a search result are word-based for exactly that reason.
 *
 * Shared rather than frontend-only because the server renders the same
 * paragraphs into the page a crawler reads, and React then renders them again;
 * two splitting rules would reflow the article under the reader on hydration.
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

export function countWords(text: string): number {
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
