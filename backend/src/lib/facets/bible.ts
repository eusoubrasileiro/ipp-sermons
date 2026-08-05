import { parseCsv } from "../corpus.ts";
import { fold, slugify } from "./slugify.ts";

/**
 * The 66-book canon, in canonical order, as the scripture facet needs it.
 *
 * Canonical order is the whole point of the table: a scripture index sorted
 * alphabetically ("Amós, Apocalipse, Atos, Cantares…") is unusable to anyone
 * who knows the Bible. `order` is the sort key everywhere, never `name`.
 *
 * Loaded from `data/facets/bible_books.csv` rather than hard-coded so the
 * spellings and abbreviations stay reviewable in git alongside the corpus.
 */
export type BibleBook = {
  order: number;
  slug: string;
  name: string;
  testament: "AT" | "NT";
  chapters: number;
  abbrevs: string[];
  aliases: string[];
};

const splitList = (v: string | undefined): string[] =>
  (v ?? "")
    .split("|")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

export function loadBibleBooks(csvText: string): BibleBook[] {
  return parseCsv(csvText)
    .map((r) => ({
      order: Number.parseInt((r.order ?? "").trim(), 10),
      slug: (r.slug ?? "").trim(),
      name: (r.name ?? "").trim(),
      testament: (r.testament ?? "").trim() === "NT" ? ("NT" as const) : ("AT" as const),
      chapters: Number.parseInt((r.chapters ?? "").trim(), 10),
      abbrevs: splitList(r.abbrevs),
      aliases: splitList(r.aliases),
    }))
    .sort((a, b) => a.order - b.order);
}

/**
 * Rewrites the ways a preacher writes a numbered book into the "1 João" form.
 *
 * The corpus contains all of these for the same book: "1 João", "I João",
 * "1º João", "primeira carta de João". Normalising here means the lookup table
 * carries one spelling per book instead of four.
 *
 * Roman numerals are converted unconditionally rather than only before a known
 * book: "I Congresso Peregrinos" becomes "1 congresso peregrinos", which
 * matches no book and is therefore harmless.
 */
function normalizeNumerals(text: string): string {
  return (
    text
      // "1º"/"1ª"/"1°" -> "1"
      .replace(/(\d)\s*[ºª°]/g, "$1")
      // "primeira carta de João" / "segunda epistola de Pedro"
      .replace(/\bprimeir[ao]\s+(?:carta|epistola|livro)\s+(?:de\s+|dos\s+|d[oa]\s+)?/g, "1 ")
      .replace(/\bsegund[ao]\s+(?:carta|epistola|livro)\s+(?:de\s+|dos\s+|d[oa]\s+)?/g, "2 ")
      .replace(/\bterceir[ao]\s+(?:carta|epistola|livro)\s+(?:de\s+|dos\s+|d[oa]\s+)?/g, "3 ")
      // Roman numerals. Longest first, or "iii" would match the "i" rule.
      .replace(/\biii\s+/g, "3 ")
      .replace(/\bii\s+/g, "2 ")
      .replace(/\bi\s+/g, "1 ")
  );
}

const escapeRegex = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

type Pattern = { re: RegExp; tier: number; length: number; book: BibleBook };

const patternCache = new WeakMap<BibleBook[], Pattern[]>();

/**
 * Full names and aliases outrank abbreviations (tier 0 vs tier 1).
 *
 * This is what resolves the one genuine collision in Portuguese: "Jó" (Job)
 * folds to "jo", which is also the standard abbreviation for João. The full
 * name wins, so "Jó 42" reads as Job and "Jo 3.16" also reads as Job. The
 * corpus has no Job sermons and spells John out, so the trade costs nothing
 * here — but it is a choice, not an accident.
 *
 * Abbreviations additionally REQUIRE a following chapter number. Without that
 * rule "Na" (Naum) matches the commonest preposition in Portuguese, and nine
 * sermons — "ensinando na palavra", "o vinho na ceia", "a Enfermidade na Vida
 * Cristã" — file themselves under a three-chapter minor prophet. An
 * abbreviation with no number after it is not a reference anyway.
 */
function buildPatterns(books: BibleBook[]): Pattern[] {
  const cached = patternCache.get(books);
  if (cached) return cached;

  const patterns: Pattern[] = [];
  for (const book of books) {
    const add = (raw: string, tier: number): void => {
      const needle = normalizeNumerals(fold(raw));
      // A cell with no letters or digits would compile to a punctuation-only
      // pattern that matches inside almost any title.
      if (!/[a-z0-9]/.test(needle)) return;
      const tail = tier === 1 ? "\\s*\\d" : "(?![a-z0-9])";
      patterns.push({
        re: new RegExp(`(?<![a-z0-9])${escapeRegex(needle)}${tail}`, "g"),
        tier,
        length: needle.length,
        book,
      });
    };
    add(book.name, 0);
    for (const alias of book.aliases) add(alias, 0);
    for (const abbrev of book.abbrevs) {
      add(abbrev, 1);
      // "1Rs" and "1 Rs" are the same abbreviation, and the corpus uses both.
      // Registering the spaced variant as its own pattern keeps the haystack a
      // faithful transform of the input, which the reference parser relies on
      // to read the chapter and verse that follow the book.
      const spaced = abbrev.match(/^([123])\s*([A-Za-zÀ-ÿ]+)$/);
      if (spaced) add(`${spaced[1]} ${spaced[2]}`, 1);
    }
  }

  patternCache.set(books, patterns);
  return patterns;
}

/**
 * Finds the book a reference string points at, or null when there is none.
 *
 * Returning null matters as much as returning a book: roughly a quarter of the
 * corpus is catechetical ("O nono mandamento", "CFW 23 - Magistrado civil",
 * "Apologética") and has no preached text at all. Anything that guesses a book
 * for those poisons the scripture index exactly where it should be empty.
 *
 * Ties break by position first, so the leading reference in a title wins over a
 * passing mention later in it.
 */
export function findBook(books: BibleBook[], text: string): BibleBook | null {
  return findBookMatch(books, text)?.book ?? null;
}

/**
 * Where the book name sits, so a caller can read the chapter and verse after it.
 *
 * `haystack` is the accent-folded, numeral-normalised text the offsets refer
 * to — never the caller's original string. Digits survive that transform
 * unchanged, which is what makes reading the reference off it safe.
 */
export type BookMatch = { book: BibleBook; index: number; end: number; haystack: string };

export function findBookMatch(books: BibleBook[], text: string): BookMatch | null {
  const haystack = normalizeNumerals(fold(text));
  if (!haystack) return null;

  let best: { index: number; tier: number; length: number; book: BibleBook } | null = null;

  for (const { re, tier, length, book } of buildPatterns(books)) {
    re.lastIndex = 0;
    const match = re.exec(haystack);
    if (!match) continue;

    const candidate = { index: match.index, tier, length, book };
    if (
      !best ||
      candidate.index < best.index ||
      (candidate.index === best.index &&
        (candidate.tier < best.tier ||
          (candidate.tier === best.tier && candidate.length > best.length)))
    ) {
      best = candidate;
    }
  }

  if (!best) return null;
  return { book: best.book, index: best.index, end: best.index + best.length, haystack };
}

/** Looks a book up by its slug, for URL params. */
export function bookBySlug(books: BibleBook[], slug: string): BibleBook | null {
  const wanted = slugify(slug);
  return books.find((b) => b.slug === wanted) ?? null;
}
