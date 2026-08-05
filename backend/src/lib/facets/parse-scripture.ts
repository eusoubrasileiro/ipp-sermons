import { type BibleBook, findBookMatch } from "./bible.ts";

/**
 * A scripture reference pulled out of a sermon title.
 *
 * `chapterStart` is null when the title names a book but no chapter ("EBD -
 * 1 Samuel"), which is common for the Sunday-school courses that walk a whole
 * book over several weeks. The book alone is still a useful facet, so a
 * chapterless reference is a result, not a failure.
 */
export type ScriptureRef = {
  bookSlug: string;
  chapterStart: number | null;
  chapterEnd: number | null;
  verseStart: number | null;
  verseEnd: number | null;
};

/**
 * A number of at most three digits, not run on into a longer one.
 *
 * The negative lookahead is what stops "Gênesis 2019" — a year that drifted
 * into a title — from being read as chapter 201.
 */
const NUM = "(\\d{1,3})(?!\\d)";
const CHAPTER_VERSE = new RegExp(`^\\s*${NUM}(?:\\s*[.:]\\s*${NUM})?`);
const RANGE_END = new RegExp(`^[a-z]?\\s*[-–—]\\s*${NUM}(?:\\s*[.:]\\s*${NUM})?`);
const MORE_VERSES = new RegExp(`^[a-z]?\\s*(?:,|\\se\\s?)\\s*${NUM}`);

const toInt = (v: string | undefined): number | null => {
  if (v === undefined) return null;
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
};

/**
 * Reads the reference a sermon title carries, or null when it carries none.
 *
 * Works on the folded haystack `findBookMatch` returns rather than the caller's
 * string, because the book name may have been rewritten on the way in ("I João"
 * -> "1 joao"). Digits are untouched by that transform, so the offsets line up.
 *
 * A chapter outside the book is dropped rather than stored: "Judas 3" is a typo
 * — Judas has one chapter — and storing it would put a sermon on a page of the
 * scripture index that can never be reached from the book listing.
 */
export function parseScriptureRef(books: BibleBook[], text: string): ScriptureRef | null {
  const match = findBookMatch(books, text);
  if (!match) return null;

  const { book, end, haystack } = match;
  const empty: ScriptureRef = {
    bookSlug: book.slug,
    chapterStart: null,
    chapterEnd: null,
    verseStart: null,
    verseEnd: null,
  };

  const head = CHAPTER_VERSE.exec(haystack.slice(end));
  if (!head) return empty;

  const chapterStart = toInt(head[1]);
  if (chapterStart === null || chapterStart < 1 || chapterStart > book.chapters) return empty;

  let verseStart = toInt(head[2]);
  let chapterEnd = chapterStart;
  let verseEnd = verseStart;

  let rest = haystack.slice(end + head[0].length);

  const range = RANGE_END.exec(rest);
  if (range) {
    const first = toInt(range[1]);
    const second = toInt(range[2]);
    if (second !== null && first !== null) {
      // "Isaías 7.17-8.8" — the range crosses into another chapter.
      chapterEnd = first;
      verseEnd = second;
    } else if (verseStart !== null && first !== null) {
      // "Efésios 5.22-33" — still chapter 5, verses 22 to 33.
      verseEnd = first;
    } else if (first !== null) {
      // "Atos 27 - 28" — a chapter range, using the same dash that separates
      // the segments of a sermon title.
      chapterEnd = first;
    }
    rest = rest.slice(range[0].length);
  }

  // "Mateus 6.12,14 e15" is one reference written three ways at once. Take the
  // last verse mentioned as the end of the span.
  for (let more = MORE_VERSES.exec(rest); more; more = MORE_VERSES.exec(rest)) {
    const v = toInt(more[1]);
    if (v === null) break;
    if (verseStart === null) verseStart = v;
    verseEnd = v;
    rest = rest.slice(more[0].length);
  }

  if (chapterEnd < chapterStart || chapterEnd > book.chapters) chapterEnd = chapterStart;

  return { bookSlug: book.slug, chapterStart, chapterEnd, verseStart, verseEnd };
}

/**
 * Every chapter a reference touches, so the scripture index can list a sermon
 * under each one.
 *
 * A Sunday-school lesson on "Gênesis 12-50" genuinely covers 39 chapters and
 * should be findable from any of them.
 */
export function chaptersOf(ref: {
  chapterStart: number | null;
  chapterEnd: number | null;
}): number[] {
  const { chapterStart, chapterEnd } = ref;
  if (chapterStart === null) return [];
  const last = chapterEnd !== null && chapterEnd > chapterStart ? chapterEnd : chapterStart;
  return Array.from({ length: last - chapterStart + 1 }, (_, i) => chapterStart + i);
}
