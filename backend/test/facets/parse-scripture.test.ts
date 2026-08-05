import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { type BibleBook, loadBibleBooks } from "../../src/lib/facets/bible.ts";
import { chaptersOf, parseScriptureRef } from "../../src/lib/facets/parse-scripture.ts";

const CSV_PATH = join(import.meta.dirname, "../../../data/facets/bible_books.csv");

describe("parseScriptureRef", () => {
  let books: BibleBook[];

  beforeAll(async () => {
    books = loadBibleBooks(await readFile(CSV_PATH, "utf8"));
  });

  const ref = (text: string) => parseScriptureRef(books, text);

  it("reads book, chapter and a verse range", () => {
    expect(ref("Efésios 5.22-33")).toEqual({
      bookSlug: "efesios",
      chapterStart: 5,
      chapterEnd: 5,
      verseStart: 22,
      verseEnd: 33,
    });
  });

  it("reads a bare chapter", () => {
    expect(ref("Tito 1")).toEqual({
      bookSlug: "tito",
      chapterStart: 1,
      chapterEnd: 1,
      verseStart: null,
      verseEnd: null,
    });
  });

  it("reads a chapter range when no verse is given", () => {
    // "Atos 27 - 28" uses the same " - " that separates title segments.
    expect(ref("Atos 27 - 28")).toMatchObject({
      bookSlug: "atos",
      chapterStart: 27,
      chapterEnd: 28,
      verseStart: null,
    });
    expect(ref("Números 15-36")).toMatchObject({ chapterStart: 15, chapterEnd: 36 });
  });

  it("reads a range that crosses a chapter boundary", () => {
    expect(ref("Isaías 7.17-8.8")).toEqual({
      bookSlug: "isaias",
      chapterStart: 7,
      chapterEnd: 8,
      verseStart: 17,
      verseEnd: 8,
    });
    expect(ref("Habacuque 1.12-2.20")).toMatchObject({ chapterStart: 1, chapterEnd: 2 });
  });

  it("ignores the verse-part letters the corpus uses", () => {
    // "Eclesiastes 3.2b-3a" — the a/b halves are display detail, not structure.
    expect(ref("Eclesiastes 3.2b-3a")).toMatchObject({
      bookSlug: "eclesiastes",
      chapterStart: 3,
      verseStart: 2,
      verseEnd: 3,
    });
    expect(ref("Efésios 6.4b")).toMatchObject({ chapterStart: 6, verseStart: 4 });
  });

  it("spans a comma-separated verse list", () => {
    // "Mateus 6.12,14 e15" is one messy reference; the span is what matters.
    expect(ref("Mateus 6.12,14 e15")).toMatchObject({
      bookSlug: "mateus",
      chapterStart: 6,
      verseStart: 12,
      verseEnd: 15,
    });
    expect(ref("Romanos 12.1,2")).toMatchObject({ verseStart: 1, verseEnd: 2 });
  });

  it("reads a numbered book and its abbreviation", () => {
    expect(ref("1 Timóteo 1.12-15")).toMatchObject({ bookSlug: "1-timoteo", chapterStart: 1 });
    expect(ref("1 Rs 14.1-20")).toMatchObject({ bookSlug: "1-reis", chapterStart: 14 });
  });

  it("reads a reference embedded in a full sermon title", () => {
    expect(ref("17-03-2024 - Efésios 5.22-33 - O casamento diante da Cruz")).toMatchObject({
      bookSlug: "efesios",
      chapterStart: 5,
      verseEnd: 33,
    });
  });

  it("does not read the date as a chapter", () => {
    // The leading "28-04-2024" must never become Atos 28.
    expect(ref("28-04-2024 - EBD - Atos 27 - 28")).toMatchObject({
      bookSlug: "atos",
      chapterStart: 27,
    });
  });

  it("returns the book with no chapter when the title only names it", () => {
    expect(ref("14/11/2021 - EBD - 1 Samuel")).toEqual({
      bookSlug: "1-samuel",
      chapterStart: null,
      chapterEnd: null,
      verseStart: null,
      verseEnd: null,
    });
  });

  it("returns null when the title has no scripture at all", () => {
    expect(ref("08-12-2019 - O Quarto Mandamento (1)")).toBeNull();
    expect(ref("21-02-2021 - EBD - Apologética")).toBeNull();
  });

  it("clamps a chapter beyond the book's length", () => {
    // "Judas 3" cannot exist — Judas has one chapter. A bad chapter must not
    // create a phantom entry in the scripture index.
    expect(ref("Judas 3")).toMatchObject({ bookSlug: "judas", chapterStart: null });
  });

  it("ignores a run-on number that is really a year", () => {
    expect(ref("Gênesis 2019")).toMatchObject({ bookSlug: "genesis", chapterStart: null });
  });
});

describe("chaptersOf", () => {
  it("expands a single chapter", () => {
    expect(chaptersOf({ chapterStart: 5, chapterEnd: 5 })).toEqual([5]);
  });

  it("expands an inclusive range", () => {
    expect(chaptersOf({ chapterStart: 27, chapterEnd: 28 })).toEqual([27, 28]);
  });

  it("returns nothing when there is no chapter", () => {
    expect(chaptersOf({ chapterStart: null, chapterEnd: null })).toEqual([]);
  });

  it("caps a runaway range rather than emitting hundreds of rows", () => {
    // "Gênesis 12-50" is legitimate and expands to 39 chapters; anything wider
    // than a book is a parse error, not a sermon.
    expect(chaptersOf({ chapterStart: 12, chapterEnd: 50 })).toHaveLength(39);
    expect(chaptersOf({ chapterStart: 5, chapterEnd: 1 })).toEqual([5]);
  });
});
