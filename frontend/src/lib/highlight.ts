/**
 * Marks the query's terms inside a transcript excerpt, and windows the excerpt
 * around the first of them.
 *
 * Matching is accent- and case-insensitive: the congregation types "ceia" and
 * the transcript says "Ceia"; someone types "efesios" and the transcript says
 * "Efésios". Postgres already folds accents on the retrieval side, so if the
 * highlighter did not, the excerpt that *caused* the hit would render plain.
 *
 * All the work happens on a folded copy of the text while slicing the original,
 * so rendered characters keep their accents.
 */

type HighlightPart = { text: string; match: boolean };

const STOPWORDS = new Set([
  "para",
  "como",
  "pelo",
  "pela",
  "que",
  "com",
  "dos",
  "das",
  "uma",
  "por",
  "nao",
  "sobre",
]);

/** Lowercases and strips diacritics so "Efésios" and "efesios" compare equal. */
export function fold(text: string): string {
  return text
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase();
}

/** Query terms worth marking: folded, de-duplicated, no filler words. */
export function queryTerms(query: string): string[] {
  const terms = fold(query)
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length > 2 && !STOPWORDS.has(t));
  return [...new Set(terms)];
}

/**
 * Folds per character so the folded copy stays index-aligned with the original
 * (NFD on the whole string would shift every offset). Returns null when the
 * alignment cannot be guaranteed, and callers fall back to plain text.
 */
function foldAligned(chars: string[]): string[] | null {
  const folded = chars.map((ch) =>
    ch
      .normalize("NFD")
      .replace(/\p{Diacritic}/gu, "")
      .toLowerCase(),
  );
  return folded.every((f) => f.length === 1) ? folded : null;
}

/** Per-character mask of which positions belong to a matched term. */
function markPositions(folded: string, terms: string[], length: number): boolean[] {
  const marked = new Array<boolean>(length).fill(false);
  for (const term of terms) {
    let from = 0;
    while (from <= folded.length - term.length) {
      const at = folded.indexOf(term, from);
      if (at === -1) break;
      // Word-prefix match only: "fé" should not light up the "fe" in "referência".
      const before = at === 0 ? "" : folded[at - 1];
      if (!before || !/[\p{L}\p{N}]/u.test(before)) {
        for (let i = at; i < at + term.length; i++) marked[i] = true;
      }
      from = at + term.length;
    }
  }
  return marked;
}

/**
 * Splits `text` into alternating plain and matched parts.
 *
 * Returns a single unmatched part when there is nothing to mark, so callers can
 * render the result unconditionally.
 */
export function highlight(text: string, query: string): HighlightPart[] {
  const terms = queryTerms(query);
  if (terms.length === 0) return [{ text, match: false }];

  const chars = [...text];
  const folded = foldAligned(chars);
  if (!folded) return [{ text, match: false }];

  const marked = markPositions(folded.join(""), terms, chars.length);

  const parts: HighlightPart[] = [];
  for (let i = 0; i < chars.length; i++) {
    const match = marked[i] === true;
    const last = parts[parts.length - 1];
    if (last && last.match === match) last.text += chars[i];
    else parts.push({ text: chars[i] ?? "", match });
  }
  return parts;
}

/**
 * A window of `text` centred on the first matched term.
 *
 * Chunks are ~1000 characters and the card previews four lines of them, so
 * clamping from the start routinely hides the very words the visitor searched
 * for -- the preview then looks like an arbitrary paragraph. Cutting on word
 * boundaries keeps it readable; ellipses mark where text was dropped.
 */
export function snippet(text: string, query: string, before = 70, after = 300): string {
  if (text.length <= before + after) return text;

  const terms = queryTerms(query);
  const chars = [...text];
  const folded = terms.length > 0 ? foldAligned(chars) : null;
  const at = folded ? markPositions(folded.join(""), terms, chars.length).indexOf(true) : -1;
  if (at === -1) return text;

  // Deliberately lopsided: a phone shows roughly four 45-character lines, so
  // only a little context can precede the match without pushing it out of the
  // clamp. The rest of the window goes after it.
  let start = Math.max(0, at - before);
  let end = Math.min(chars.length, at + after);
  if (start > 0) {
    const space = chars.indexOf(" ", start);
    if (space !== -1 && space < at) start = space + 1;
  }
  if (end < chars.length) {
    const space = chars.lastIndexOf(" ", end);
    if (space > at) end = space;
  }

  return `${start > 0 ? "…" : ""}${chars.slice(start, end).join("")}${end < chars.length ? "…" : ""}`;
}
