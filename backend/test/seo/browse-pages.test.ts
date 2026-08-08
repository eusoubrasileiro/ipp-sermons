import type { PrismaClient } from "@prisma/client";
import { describe, expect, it, vi } from "vitest";
import { facetIndexPage, facetLeafPage, homePage } from "../../src/lib/seo/browse-pages.ts";

/**
 * The crawl path.
 *
 * Every sermon URL is an island unless something links to it in HTML, and these
 * pages are that something. What is pinned here is mostly refusal: a slug the
 * archive does not have, a chapter that cannot exist, a month that is not a
 * month — each has to come back as null so the route falls through to the SPA
 * rather than publishing an indexable page about nothing.
 */

const BOOK_ROWS = [
  { slug: "genesis", nome: "Gênesis", testamento: "AT", ordem: 1, chapter: 1, total: 3 },
  { slug: "genesis", nome: "Gênesis", testamento: "AT", ordem: 1, chapter: 17, total: 2 },
  { slug: "tito", nome: "Tito", testamento: "NT", ordem: 56, chapter: 2, total: 1 },
];

/**
 * `facetTree()` issues eight GROUP BYs; each is recognisable by the table it
 * reads, so one stub answers all of them without pretending to be Postgres.
 */
const facetRows = async (strings: TemplateStringsArray) => {
  const sql = Array.isArray(strings) ? strings.join(" ") : String(strings);
  if (sql.includes("FROM bible_books")) return BOOK_ROWS;
  if (sql.includes("FROM sermon_scriptures GROUP BY")) {
    return [
      { slug: "genesis", total: 4 },
      { slug: "tito", total: 1 },
    ];
  }
  if (sql.includes("FROM series se")) {
    return [
      {
        slug: "o-livro-dos-reis",
        nome: "O livro dos Reis",
        kind: "curso",
        pai: null,
        descricao: "",
        total: 7,
      },
      {
        slug: "serie-vazia",
        nome: "Série vazia",
        kind: "curso",
        pai: null,
        descricao: "",
        total: 0,
      },
    ];
  }
  if (sql.includes("FROM sermons GROUP BY artist")) {
    return [
      { artist: "Reverendo Bruno Melo", total: 300 },
      { artist: "Pastor Lucas Antunes", total: 40 },
    ];
  }
  if (sql.includes("FROM sermons GROUP BY 1, 2")) {
    return [
      { ano: 2024, mes: 3, total: 4 },
      { ano: 2023, mes: 12, total: 2 },
    ];
  }
  if (sql.includes("FROM topics t")) {
    return [
      { slug: "graca", nome: "Graça", grupo: "deus", grupoNome: "Deus", total: 9 },
      { slug: "sem-uso", nome: "Sem uso", grupo: "deus", grupoNome: "Deus", total: 0 },
    ];
  }
  return [];
};

const sermonRows = [
  {
    id: "111",
    title: "18-07-2021 - Gênesis 17.9-27",
    artist: "Reverendo Bruno Melo",
    date: new Date("2021-07-18T00:00:00Z"),
  },
];

const stubPrisma = () =>
  ({
    $queryRaw: vi.fn(facetRows),
    sermon: {
      count: vi.fn(async () => 560),
      findMany: vi.fn(async () => sermonRows),
    },
  }) as unknown as PrismaClient;

describe("facetIndexPage", () => {
  it("links every book of the canon it has sermons for", async () => {
    const page = await facetIndexPage(stubPrisma(), "biblia");

    expect(page.path).toBe("/biblia");
    expect(page.body).toContain('href="/biblia/genesis"');
    expect(page.body).toContain('href="/biblia/tito"');
  });

  it("leaves out a topic and a series nothing is filed under", async () => {
    const prisma = stubPrisma();

    const temas = await facetIndexPage(prisma, "temas");
    expect(temas.body).toContain('href="/temas/graca"');
    expect(temas.body).not.toContain("sem-uso");

    const series = await facetIndexPage(prisma, "series");
    expect(series.body).toContain('href="/series/o-livro-dos-reis"');
    expect(series.body).not.toContain("serie-vazia");
  });

  it("names a preacher the way the filter does, honorific and all", async () => {
    // `artist` is the raw column and the only value a filter matches on;
    // splitting it for display here would produce a name that matches nothing.
    const page = await facetIndexPage(stubPrisma(), "pregadores");
    expect(page.body).toContain("Reverendo Bruno Melo");
  });

  it("lists the years newest first", async () => {
    const page = await facetIndexPage(stubPrisma(), "datas");
    expect(page.body.indexOf("/datas/2024")).toBeLessThan(page.body.indexOf("/datas/2023"));
  });
});

describe("facetLeafPage", () => {
  it("resolves a book to its canonical name and filters on its slug", async () => {
    const prisma = stubPrisma();
    const page = await facetLeafPage(prisma, "biblia", "genesis");

    expect(page?.title).toContain("Gênesis");
    expect(page?.path).toBe("/biblia/genesis");
    expect(page?.body).toContain('href="/sermao/111"');
    expect(
      JSON.stringify((prisma.sermon.count as unknown as { mock: { calls: unknown[] } }).mock.calls),
    ).toContain("genesis");
  });

  it("narrows to a chapter when the url names one", async () => {
    const prisma = stubPrisma();
    const page = await facetLeafPage(prisma, "biblia", "genesis", "17");

    expect(page?.path).toBe("/biblia/genesis/17");
    expect(page?.title).toContain("Gênesis 17");
    expect(
      JSON.stringify((prisma.sermon.count as unknown as { mock: { calls: unknown[] } }).mock.calls),
    ).toContain('"chapter":17');
  });

  it("refuses a chapter no book has", async () => {
    expect(await facetLeafPage(stubPrisma(), "biblia", "genesis", "999")).toBe(null);
    expect(await facetLeafPage(stubPrisma(), "biblia", "genesis", "zero")).toBe(null);
  });

  it("refuses a slug the archive does not have", async () => {
    const prisma = stubPrisma();
    expect(await facetLeafPage(prisma, "biblia", "nao-existe")).toBe(null);
    expect(await facetLeafPage(prisma, "temas", "nao-existe")).toBe(null);
    expect(await facetLeafPage(prisma, "series", "nao-existe")).toBe(null);
    expect(await facetLeafPage(prisma, "pregadores", "nao-existe")).toBe(null);
  });

  it("resolves a topic, a series and a preacher", async () => {
    const prisma = stubPrisma();

    expect((await facetLeafPage(prisma, "temas", "graca"))?.title).toContain("Graça");
    expect((await facetLeafPage(prisma, "series", "o-livro-dos-reis"))?.title).toContain(
      "O livro dos Reis",
    );

    const preacher = await facetLeafPage(prisma, "pregadores", "reverendo-bruno-melo");
    expect(preacher?.title).toContain("Reverendo Bruno Melo");
  });

  it("refuses a third path segment on a facet that has no third level", async () => {
    const prisma = stubPrisma();
    expect(await facetLeafPage(prisma, "temas", "graca", "2")).toBe(null);
    expect(await facetLeafPage(prisma, "series", "o-livro-dos-reis", "2")).toBe(null);
  });

  it("turns a year into a date range rather than looking it up", async () => {
    const prisma = stubPrisma();
    const page = await facetLeafPage(prisma, "datas", "2024");

    expect(page?.path).toBe("/datas/2024");
    const where = JSON.stringify(
      (prisma.sermon.count as unknown as { mock: { calls: unknown[] } }).mock.calls,
    );
    expect(where).toContain("2024-01-01");
    expect(where).toContain("2024-12-31");
  });

  it("ends a month on its own last day, not on the 31st of every month", async () => {
    // February is the case that catches a naive `${month}-31`.
    const prisma = stubPrisma();
    const page = await facetLeafPage(prisma, "datas", "2024", "2");

    expect(page?.title).toContain("fevereiro de 2024");
    expect(
      JSON.stringify((prisma.sermon.count as unknown as { mock: { calls: unknown[] } }).mock.calls),
    ).toContain("2024-02-29");
  });

  it("refuses anything that is not a year or a month", async () => {
    const prisma = stubPrisma();
    expect(await facetLeafPage(prisma, "datas", "nao-e-um-ano")).toBe(null);
    expect(await facetLeafPage(prisma, "datas", "1500")).toBe(null);
    expect(await facetLeafPage(prisma, "datas", "2024", "13")).toBe(null);
    expect(await facetLeafPage(prisma, "datas", "2024", "marco")).toBe(null);
  });

  it("counts in Portuguese, singular and plural", async () => {
    const prisma = {
      $queryRaw: vi.fn(facetRows),
      sermon: { count: vi.fn(async () => 1), findMany: vi.fn(async () => sermonRows) },
    } as unknown as PrismaClient;

    expect((await facetLeafPage(prisma, "biblia", "tito"))?.description).toContain("1 sermão em");
  });
});

describe("homePage", () => {
  it("links into every index and says how large the archive is", async () => {
    const page = await homePage(stubPrisma());

    expect(page.path).toBe("/");
    expect(page.ogType).toBe("website");
    expect(page.description).toContain("560");
    for (const family of ["biblia", "temas", "series", "pregadores", "datas"]) {
      expect(page.body).toContain(`href="/${family}"`);
    }
  });
});
