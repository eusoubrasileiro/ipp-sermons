import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import {
  type BibleBook,
  type BookMatch,
  bookBySlug,
  findBook,
  findBookMatch,
  loadBibleBooks,
} from "../../src/lib/facets/bible.ts";
import { slugify } from "../../src/lib/facets/slugify.ts";

const CSV_PATH = join(import.meta.dirname, "../../../data/facets/bible_books.csv");

describe("slugify", () => {
  it("folds Portuguese accents", () => {
    expect(slugify("Gênesis")).toBe("genesis");
    expect(slugify("Efésios")).toBe("efesios");
    expect(slugify("Confissão de Fé")).toBe("confissao-de-fe");
  });

  it("keeps leading numerals, which distinguish books", () => {
    // "1 Reis" and "2 Reis" are different books; dropping the digit merges them.
    expect(slugify("1 Reis")).toBe("1-reis");
    expect(slugify("2 Timóteo")).toBe("2-timoteo");
  });

  it("collapses punctuation and repeated separators", () => {
    expect(slugify("Presbíteros, Bispos e Pastores")).toBe("presbiteros-bispos-e-pastores");
    expect(slugify("  EBD  --  Atos  ")).toBe("ebd-atos");
  });

  it("is idempotent", () => {
    expect(slugify(slugify("O Livro dos Reis"))).toBe("o-livro-dos-reis");
  });

  it("returns an empty string for input with nothing sluggable", () => {
    expect(slugify("   ---   ")).toBe("");
  });
});

describe("bible_books.csv", () => {
  let books: BibleBook[];

  beforeAll(async () => {
    books = loadBibleBooks(await readFile(CSV_PATH, "utf8"));
  });

  it("has all 66 books in canonical order", () => {
    expect(books).toHaveLength(66);
    expect(books[0]?.slug).toBe("genesis");
    expect(books[65]?.slug).toBe("apocalipse");
    expect(books.map((b) => b.order)).toEqual(books.map((_, i) => i + 1));
  });

  it("splits the testaments at Malaquias / Mateus", () => {
    expect(books.filter((b) => b.testament === "AT")).toHaveLength(39);
    expect(books.filter((b) => b.testament === "NT")).toHaveLength(27);
    expect(books[38]?.slug).toBe("malaquias");
    expect(books[39]?.slug).toBe("mateus");
  });

  it("gives every book a plausible chapter count", () => {
    expect(books.find((b) => b.slug === "salmos")?.chapters).toBe(150);
    expect(books.find((b) => b.slug === "judas")?.chapters).toBe(1);
    expect(books.every((b) => b.chapters >= 1 && b.chapters <= 150)).toBe(true);
  });

  it("has unique slugs", () => {
    expect(new Set(books.map((b) => b.slug)).size).toBe(66);
  });

  it("survives a CSV that is missing the optional columns", () => {
    // `abbrevs` and `aliases` are empty for most books and a hand-edited file
    // can drop them entirely; that must not throw on every row.
    const sparse = loadBibleBooks("order,slug,name,testament,chapters\n1,tito,Tito,NT,3\n");
    expect(sparse[0]).toMatchObject({ slug: "tito", abbrevs: [], aliases: [] });
  });

  it("defaults a row with nothing usable rather than throwing", () => {
    const empty = loadBibleBooks("nada\nx\n");
    expect(empty[0]?.testament).toBe("AT");
    expect(empty[0]?.slug).toBe("");
  });
});

describe("findBook", () => {
  let books: BibleBook[];

  beforeAll(async () => {
    books = loadBibleBooks(await readFile(CSV_PATH, "utf8"));
  });

  const slugOf = (text: string) => findBook(books, text)?.slug ?? null;

  it("matches the canonical name", () => {
    expect(slugOf("Gênesis")).toBe("genesis");
    expect(slugOf("Apocalipse")).toBe("apocalipse");
  });

  it("matches without accents, as people type", () => {
    expect(slugOf("genesis")).toBe("genesis");
    expect(slugOf("EFESIOS")).toBe("efesios");
  });

  it("matches the abbreviations used in sermon titles", () => {
    // "1 Rs 14.1-20" appears verbatim in the corpus.
    expect(slugOf("1 Rs 14.1-20")).toBe("1-reis");
    expect(slugOf("Ef 5")).toBe("efesios");
  });

  it("accepts roman numerals for the numbered books", () => {
    // The corpus has a transcript named "I João 3.19-24".
    expect(slugOf("I João")).toBe("1-joao");
    expect(slugOf("II Samuel")).toBe("2-samuel");
    expect(slugOf("III João")).toBe("3-joao");
  });

  it("accepts the ordinal forms a preacher writes", () => {
    expect(slugOf("1º Coríntios")).toBe("1-corintios");
    expect(slugOf("primeira carta de João")).toBe("1-joao");
  });

  it("accepts the singular 'Salmo', which the corpus uses", () => {
    expect(slugOf("Salmo 23")).toBe("salmos");
    expect(slugOf("Salmos 23")).toBe("salmos");
  });

  it("finds a book inside a longer reference string", () => {
    expect(slugOf("Efésios 5.22-33")).toBe("efesios");
    expect(slugOf("28-04-2024 - EBD - Atos 27 - 28")).toBe("atos");
  });

  it("prefers the numbered book over the bare one", () => {
    // "1 João" must not resolve to the Gospel of João.
    expect(slugOf("1 João 3.19-24")).toBe("1-joao");
    expect(slugOf("2 Pedro 1")).toBe("2-pedro");
    expect(slugOf("1 Timóteo 1.12-15")).toBe("1-timoteo");
  });

  it("returns null when there is no book at all", () => {
    // Real corpus titles: catechetical, with no preached text.
    expect(slugOf("O Quarto Mandamento")).toBeNull();
    expect(slugOf("EBD - Apologética")).toBeNull();
    expect(slugOf("I Congresso Peregrinos - Santificação")).toBeNull();
  });

  it("does not match a book name embedded in an unrelated word", () => {
    expect(slugOf("Joãozinho")).toBeNull();
  });

  it("takes the leading reference when a title mentions two books", () => {
    expect(slugOf("Salmo 23 e João 10")).toBe("salmos");
  });

  it("lets the full name beat an abbreviation starting at the same spot", () => {
    // "Jó" and João's abbreviation "Jo" both fold to "jo" and both match at
    // index 0 of "jo 42". The full name is the more literal read and wins.
    expect(slugOf("Jó 42")).toBe("jo");
  });

  it("prefers the longer of two aliases anchored at the same spot", () => {
    expect(slugOf("Atos dos Apóstolos 2")).toBe("atos");
  });

  it("is stable when two spellings of a name are the same length", () => {
    // The CSV carries both "Oseias" and "Oséias"; they fold to the same string.
    expect(slugOf("Oseias 3")).toBe("oseias");
    expect(slugOf("Oséias 3")).toBe("oseias");
  });

  it("returns null for empty or unsluggable input", () => {
    expect(slugOf("")).toBeNull();
    expect(slugOf("   ")).toBeNull();
    expect(slugOf("--- ---")).toBeNull();
  });

  it("ignores a malformed abbreviation rather than matching everything", () => {
    // A punctuation-only cell would otherwise compile to an empty pattern,
    // which matches at index 0 of every string.
    const broken = loadBibleBooks(
      "order,slug,name,testament,chapters,abbrevs,aliases\n1,tito,Tito,NT,3,--|Tt,\n",
    );
    expect(findBook(broken, "Tito 2")?.slug).toBe("tito");
    expect(findBook(broken, "O nono mandamento")).toBeNull();
  });
});

describe("findBookMatch", () => {
  let books: BibleBook[];

  beforeAll(async () => {
    books = loadBibleBooks(await readFile(CSV_PATH, "utf8"));
  });

  it("reports the span the book name occupies in the folded text", () => {
    const match: BookMatch | null = findBookMatch(books, "EBD - Efésios 5.22-33");
    expect(match?.book.slug).toBe("efesios");
    expect(match?.haystack.slice(match.index, match.end)).toBe("efesios");
    // What follows the span is the reference the scripture parser reads.
    expect(match?.haystack.slice(match.end)).toBe(" 5.22-33");
  });

  it("reports the span against the rewritten text, not the caller's", () => {
    // "I João" is normalised to "1 joao" before matching, so the offsets are
    // into the normalised haystack — which is why it is returned alongside.
    const match = findBookMatch(books, "I João 3.19-24");
    expect(match?.haystack.slice(match.index, match.end)).toBe("1 joao");
  });

  it("is null when there is no book", () => {
    expect(findBookMatch(books, "O Quarto Mandamento")).toBeNull();
  });
});

describe("bookBySlug", () => {
  let books: BibleBook[];

  beforeAll(async () => {
    books = loadBibleBooks(await readFile(CSV_PATH, "utf8"));
  });

  it("resolves a URL param", () => {
    expect(bookBySlug(books, "1-timoteo")?.name).toBe("1 Timóteo");
  });

  it("tolerates an unslugged param", () => {
    expect(bookBySlug(books, "Efésios")?.slug).toBe("efesios");
  });

  it("returns null for an unknown slug", () => {
    expect(bookBySlug(books, "evangelho-de-tome")).toBeNull();
  });
});
