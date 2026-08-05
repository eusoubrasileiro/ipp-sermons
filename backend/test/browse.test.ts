import type { PrismaClient } from "@prisma/client";
import { describe, expect, it, vi } from "vitest";
import { type FacetTree, facetTree, splitPreacher } from "../src/lib/browse/facets.ts";
import { listSermons } from "../src/lib/browse/list.ts";

describe("splitPreacher", () => {
  it("separates the honorific from the person", () => {
    expect(splitPreacher("Reverendo Bruno Melo")).toEqual({
      titulo: "Reverendo",
      nome: "Bruno Melo",
    });
    expect(splitPreacher("Presbítero Humberto Elias")).toEqual({
      titulo: "Presbítero",
      nome: "Humberto Elias",
    });
  });

  it("handles every honorific the corpus uses", () => {
    for (const titulo of ["Pastor", "Seminarista", "Diácono", "Missionário"]) {
      expect(splitPreacher(`${titulo} Fulano de Tal`).titulo).toBe(titulo);
    }
  });

  it("keeps a name that carries no honorific", () => {
    // One entry in preacher_names.txt has no title at all.
    expect(splitPreacher("Paula Ximenes")).toEqual({ titulo: "Outros", nome: "Paula Ximenes" });
    expect(splitPreacher("Desconhecido")).toEqual({ titulo: "Outros", nome: "Desconhecido" });
  });

  it("keeps a multi-word surname intact", () => {
    expect(splitPreacher("Reverendo Alan Rennê Alexandrino").nome).toBe("Alan Rennê Alexandrino");
  });
});

describe("facetTree", () => {
  it("always returns every family, even on an empty corpus", async () => {
    // An index page that renders nothing is fine; one that crashes because a
    // family is missing is not, and the topics family is empty until the
    // labelling pass has run.
    const prisma = { $queryRaw: vi.fn(async () => []) } as unknown as PrismaClient;
    const tree: FacetTree = await facetTree(prisma);

    expect(tree.livros).toEqual([]);
    expect(tree.series).toEqual([]);
    expect(tree.pregadores).toEqual([]);
    expect(tree.datas).toEqual([]);
    expect(tree.tipos).toEqual([]);
    expect(tree.temas).toEqual([]);
  });

  it("groups months under their year, newest first", async () => {
    const prisma = {
      $queryRaw: vi.fn(async (strings: TemplateStringsArray) => {
        const sql = strings.join(" ");
        if (!sql.includes("EXTRACT(YEAR")) return [];
        return [
          { ano: 2024, mes: 3, total: 5n },
          { ano: 2024, mes: 1, total: 4n },
          { ano: 2023, mes: 12, total: 2n },
        ];
      }),
    } as unknown as PrismaClient;

    const { datas } = await facetTree(prisma);
    expect(datas.map((d) => d.ano)).toEqual([2024, 2023]);
    expect(datas[0]?.total).toBe(9);
    expect(datas[0]?.meses).toHaveLength(2);
  });
});

/** Routes each facet query to its own canned rows, keyed on the SQL text. */
const stubQueries = (byFragment: Record<string, unknown[]>) =>
  ({
    $queryRaw: vi.fn(async (strings: TemplateStringsArray) => {
      const sql = strings.join(" ");
      for (const [fragment, rows] of Object.entries(byFragment)) {
        if (sql.includes(fragment)) return rows;
      }
      return [];
    }),
  }) as unknown as PrismaClient;

describe("facetTree — mapping", () => {
  it("counts a book by distinct sermons, not by its chapters", async () => {
    // One lesson on "Gênesis 12-50" contributes 39 chapter rows and one sermon.
    // Summing the chapters would report 39 sermons on Genesis.
    const prisma = stubQueries({
      "JOIN sermon_scriptures ss": [
        { slug: "genesis", nome: "Gênesis", testamento: "AT", ordem: 1, chapter: 12, total: 1n },
        { slug: "genesis", nome: "Gênesis", testamento: "AT", ordem: 1, chapter: 13, total: 1n },
      ],
      "GROUP BY book_slug": [{ slug: "genesis", total: 1n }],
    });

    const { livros } = await facetTree(prisma);
    expect(livros[0]?.total).toBe(1);
    expect(livros[0]?.capitulos).toHaveLength(2);
  });

  it("drops chapter 0 from the browsable chapters", async () => {
    // 0 means the title named the book with no chapter — real for the book
    // count, but not a page anyone can navigate to.
    const prisma = stubQueries({
      "JOIN sermon_scriptures ss": [
        { slug: "1-samuel", nome: "1 Samuel", testamento: "AT", ordem: 9, chapter: 0, total: 3n },
        { slug: "1-samuel", nome: "1 Samuel", testamento: "AT", ordem: 9, chapter: 4, total: 1n },
      ],
      "GROUP BY book_slug": [{ slug: "1-samuel", total: 3n }],
    });

    const { livros } = await facetTree(prisma);
    expect(livros[0]?.total).toBe(3);
    expect(livros[0]?.capitulos).toEqual([{ numero: 4, total: 1 }]);
  });

  it("returns books in canonical order, never alphabetical", async () => {
    const prisma = stubQueries({
      "JOIN sermon_scriptures ss": [
        {
          slug: "apocalipse",
          nome: "Apocalipse",
          testamento: "NT",
          ordem: 66,
          chapter: 1,
          total: 1n,
        },
        { slug: "genesis", nome: "Gênesis", testamento: "AT", ordem: 1, chapter: 1, total: 1n },
      ],
      "GROUP BY book_slug": [],
    });

    const { livros } = await facetTree(prisma);
    expect(livros.map((b) => b.slug)).toEqual(["genesis", "apocalipse"]);
  });

  it("maps series, preachers, types and topics", async () => {
    const prisma = stubQueries({
      "FROM series se": [
        {
          slug: "diaconia",
          nome: "Diaconia",
          kind: "diaconia",
          pai: null,
          descricao: "d",
          total: 7n,
        },
      ],
      "FROM sermons GROUP BY artist": [{ artist: "Reverendo Bruno Melo", total: 268n }],
      "WHERE service_type IS NOT NULL": [{ slug: "culto", total: 257n }],
      "FROM topics t": [
        {
          slug: "ansiedade",
          nome: "Ansiedade",
          grupo: "vida-crista",
          grupoNome: "Vida Cristã",
          total: 4n,
        },
      ],
    });

    const tree = await facetTree(prisma);
    expect(tree.series[0]).toMatchObject({ slug: "diaconia", total: 7 });
    expect(tree.pregadores[0]).toMatchObject({
      nome: "Bruno Melo",
      titulo: "Reverendo",
      total: 268,
    });
    expect(tree.tipos[0]).toMatchObject({ slug: "culto", total: 257 });
    expect(tree.temas[0]).toMatchObject({ slug: "ansiedade", grupoNome: "Vida Cristã", total: 4 });
  });
});

describe("listSermons", () => {
  const spy = () => {
    const prisma = {
      sermon: { count: vi.fn(async () => 0), findMany: vi.fn(async () => []) },
    } as unknown as PrismaClient;
    return prisma;
  };
  const whereOf = (prisma: PrismaClient) =>
    (prisma.sermon.count as unknown as { mock: { calls: { where: unknown }[][] } }).mock
      .calls[0]?.[0]?.where;

  it("builds no filter at all when none is given", async () => {
    const prisma = spy();
    await listSermons(prisma, {});
    expect(whereOf(prisma)).toEqual({});
  });

  it("filters by preacher, type and series", async () => {
    const prisma = spy();
    await listSermons(prisma, { pregadores: ["X"], tipos: ["ebd"], series: ["diaconia"] });
    expect(whereOf(prisma)).toMatchObject({
      artist: { in: ["X"] },
      serviceType: { in: ["ebd"] },
      seriesSlug: { in: ["diaconia"] },
    });
  });

  it("filters by book and chapter through the join table", async () => {
    const prisma = spy();
    await listSermons(prisma, { livros: ["efesios"], capitulo: 5 });
    expect(JSON.stringify(whereOf(prisma))).toContain('"chapter":5');
  });

  it("filters by topic", async () => {
    const prisma = spy();
    await listSermons(prisma, { temas: ["ansiedade"] });
    expect(JSON.stringify(whereOf(prisma))).toContain("ansiedade");
  });

  it("accepts an open-ended date range", async () => {
    const prisma = spy();
    await listSermons(prisma, { de: "2024-01-01" });
    expect(whereOf(prisma)).toHaveProperty("date");
  });

  it("orders a course by its lesson number, introduction first", async () => {
    const prisma = spy();
    await listSermons(prisma, { series: ["o-livro-dos-reis"] }, "serie");
    const args = (prisma.sermon.findMany as unknown as { mock: { calls: unknown[][] } }).mock
      .calls[0]?.[0];
    expect(JSON.stringify(args)).toContain('"nulls":"first"');
  });
});
