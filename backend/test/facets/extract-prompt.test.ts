import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadBibleBooks } from "../../src/lib/facets/bible.ts";
import {
  bookNames,
  decisionToRows,
  openingWords,
  WINDOW_WORDS,
} from "../../src/lib/facets/extract-prompt.ts";

/**
 * The window and the answer shape for the LLM scripture pass.
 *
 * Two failure modes matter and both are silent: a window too short misses the
 * passage the preacher announced in his second paragraph, and an over-eager
 * model files a catechism lesson under whatever verse it happened to quote.
 */
const books = loadBibleBooks(
  readFileSync(join(import.meta.dirname, "../../../data/facets/bible_books.csv"), "utf8"),
);

const decision = (over: Record<string, unknown> = {}) => ({
  livro: "Efésios",
  capitulo_inicio: 5,
  versiculo_inicio: 22,
  capitulo_fim: 5,
  versiculo_fim: 33,
  justificativa: "o pregador anuncia a leitura",
  ...over,
});

describe("openingWords", () => {
  it("keeps the opening of the sermon and nothing after it", () => {
    const text = Array.from({ length: 5000 }, (_, i) => `p${i}`).join(" ");
    const window = openingWords(text, 10);

    expect(window.split(/\s+/)).toHaveLength(10);
    expect(window.startsWith("p0 p1")).toBe(true);
  });

  it("takes 2500 words by default", () => {
    // Measured on this corpus: 400 words recovers 53% of the title-less
    // sermons, 2500 recovers ~97%. Anything past that is supporting citation,
    // not the text being preached.
    expect(WINDOW_WORDS).toBe(2500);
    const text = Array.from({ length: 4000 }, () => "palavra").join(" ");
    expect(openingWords(text).split(/\s+/)).toHaveLength(2500);
  });

  it("returns a shorter transcript untouched", () => {
    expect(openingWords("um sermão curto", 2500)).toBe("um sermão curto");
  });

  it("collapses the line breaks the transcripts carry", () => {
    expect(openingWords("uma\n\nfrase   quebrada", 10)).toBe("uma frase quebrada");
  });
});

describe("bookNames", () => {
  it("offers all 66 books as a closed list", () => {
    // The model picks from an enum rather than free-typing a name, so an
    // invented book cannot reach the CSV at all.
    expect(bookNames(books)).toHaveLength(66);
    expect(bookNames(books)).toContain("Gênesis");
    expect(bookNames(books)).toContain("Apocalipse");
  });
});

describe("decisionToRows", () => {
  it("turns a passage into one row per chapter", () => {
    const rows = decisionToRows(books, "s1", decision({ capitulo_fim: 6 }));

    expect(rows.map((r) => r.chapter)).toEqual([5, 6]);
    // Verses bound only the chapters they touch.
    expect(rows[0]).toMatchObject({ verse_start: 22, verse_end: null, source: "transcricao" });
    expect(rows[1]).toMatchObject({ verse_start: null, verse_end: 33 });
  });

  it("writes nothing when the sermon has no passage", () => {
    // Catechetical sermons -- "O nono mandamento", "CFW 23" -- legitimately
    // have none, and inventing one pollutes the index exactly where it should
    // be empty.
    expect(decisionToRows(books, "s1", decision({ livro: null }))).toEqual([]);
  });

  it("writes nothing for a book that does not exist", () => {
    expect(decisionToRows(books, "s1", decision({ livro: "Enoque" }))).toEqual([]);
  });

  it("files a book with no chapter under chapter 0, as the title parser does", () => {
    const rows = decisionToRows(
      books,
      "s1",
      decision({ capitulo_inicio: null, capitulo_fim: null }),
    );

    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ book_slug: "efesios", chapter: 0 });
  });

  it("refuses a chapter the book does not have", () => {
    // Naum has 3 chapters; a model that answers "Naum 12" has hallucinated,
    // and a bad row here is invisible until someone browses to an empty page.
    expect(
      decisionToRows(
        books,
        "s1",
        decision({ livro: "Naum", capitulo_inicio: 12, capitulo_fim: 12 }),
      ),
    ).toEqual([]);
  });

  it("clamps a range that runs past the end of the book", () => {
    const rows = decisionToRows(
      books,
      "s1",
      decision({ livro: "Judas", capitulo_inicio: 1, capitulo_fim: 4, versiculo_fim: null }),
    );

    expect(rows.map((r) => r.chapter)).toEqual([1]);
  });

  it("marks every row as coming from the transcript, not the title", () => {
    // The source column is what lets a later review tell a parsed reference
    // from a guessed one.
    for (const row of decisionToRows(books, "s1", decision())) {
      expect(row.source).toBe("transcricao");
      expect(row.is_primary).toBe(true);
    }
  });
});
