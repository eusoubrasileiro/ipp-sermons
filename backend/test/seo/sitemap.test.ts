import type { PrismaClient } from "@prisma/client";
import { describe, expect, it, vi } from "vitest";
import { buildRobots, buildSitemap } from "../../src/lib/seo/sitemap.ts";

/**
 * The sitemap is the only thing that tells a crawler these 560 URLs exist.
 *
 * Generated from the database on request rather than committed: the corpus
 * grows by 50–100 sermons a year through `tools/corpus-update`, and a committed
 * file is one more thing that silently goes stale between releases.
 */

const SITE = "https://exemplo.test";

/** `facetTree()`'s eight GROUP BYs, each recognisable by the table it reads. */
const facetRows = async (strings: TemplateStringsArray) => {
  const sql = Array.isArray(strings) ? strings.join(" ") : String(strings);
  if (sql.includes("FROM bible_books")) {
    return [{ slug: "tito", nome: "Tito", testamento: "NT", ordem: 56, chapter: 2, total: 1 }];
  }
  if (sql.includes("FROM sermon_scriptures GROUP BY")) return [{ slug: "tito", total: 1 }];
  if (sql.includes("FROM series se")) {
    return [
      { slug: "reis", nome: "Reis", kind: "curso", pai: null, descricao: "", total: 7 },
      { slug: "vazia", nome: "Vazia", kind: "curso", pai: null, descricao: "", total: 0 },
    ];
  }
  if (sql.includes("FROM sermons GROUP BY artist")) {
    return [{ artist: "Reverendo Bruno Melo", total: 300 }];
  }
  if (sql.includes("FROM sermons GROUP BY 1, 2")) return [{ ano: 2024, mes: 3, total: 4 }];
  if (sql.includes("FROM topics t")) {
    return [
      { slug: "graca", nome: "Graça", grupo: "deus", grupoNome: "Deus", total: 9 },
      { slug: "sem-uso", nome: "Sem uso", grupo: "deus", grupoNome: "Deus", total: 0 },
    ];
  }
  return [];
};

const stubPrisma = (over: Record<string, unknown> = {}) =>
  ({
    sermon: {
      findMany: vi.fn(async () => [
        { id: "111", date: new Date("2023-01-01T00:00:00Z") },
        { id: "222", date: new Date("2024-06-30T00:00:00Z") },
      ]),
    },
    $queryRaw: vi.fn(facetRows),
    ...over,
  }) as unknown as PrismaClient;

describe("buildSitemap", () => {
  it("lists one url per sermon, with the date it was preached as lastmod", async () => {
    const xml = await buildSitemap(stubPrisma(), SITE);

    expect(xml).toContain("<loc>https://exemplo.test/sermao/111</loc>");
    expect(xml).toContain("<loc>https://exemplo.test/sermao/222</loc>");
    expect(xml).toContain("<lastmod>2024-06-30</lastmod>");
  });

  it("is a well-formed urlset a crawler will accept", async () => {
    const xml = await buildSitemap(stubPrisma(), SITE);

    expect(xml.startsWith('<?xml version="1.0" encoding="UTF-8"?>')).toBe(true);
    expect(xml).toContain('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">');
    expect(xml.trimEnd().endsWith("</urlset>")).toBe(true);
  });

  it("includes the browse indexes, which are the crawl paths into the corpus", async () => {
    const xml = await buildSitemap(stubPrisma(), SITE);

    for (const path of ["/", "/biblia", "/temas", "/series", "/pregadores", "/datas"]) {
      expect(xml).toContain(`<loc>https://exemplo.test${path}</loc>`);
    }
  });

  it("lists every facet leaf, down to the chapter and the year", async () => {
    const xml = await buildSitemap(stubPrisma(), SITE);

    expect(xml).toContain("<loc>https://exemplo.test/biblia/tito</loc>");
    expect(xml).toContain("<loc>https://exemplo.test/biblia/tito/2</loc>");
    expect(xml).toContain("<loc>https://exemplo.test/temas/graca</loc>");
    expect(xml).toContain("<loc>https://exemplo.test/series/reis</loc>");
    expect(xml).toContain("<loc>https://exemplo.test/pregadores/reverendo-bruno-melo</loc>");
    expect(xml).toContain("<loc>https://exemplo.test/datas/2024</loc>");
  });

  it("does not advertise a facet with nothing under it", async () => {
    // A url that lists no sermons is a soft 404 to a crawler, and enough of
    // them cost the whole site crawl budget.
    const xml = await buildSitemap(stubPrisma(), SITE);

    expect(xml).not.toContain("/temas/sem-uso");
    expect(xml).not.toContain("/series/vazia");
  });

  it("escapes an id that would otherwise break the XML", async () => {
    // Six sermons predate SoundCloud and fall back to their title as an id.
    const prisma = stubPrisma({
      sermon: {
        findMany: vi.fn(async () => [
          { id: "Tiago & Filipenses", date: new Date("2020-01-05T00:00:00Z") },
        ]),
      },
    });
    const xml = await buildSitemap(prisma, SITE);

    // Percent-encoded before it is XML-escaped, so the ampersand never reaches
    // the document as a character an XML parser has to interpret.
    expect(xml).not.toContain("Tiago & Filipenses");
    expect(xml).toContain("<loc>https://exemplo.test/sermao/Tiago%20%26%20Filipenses</loc>");
  });
});

describe("buildRobots", () => {
  it("points at the sitemap and keeps crawlers out of the API", () => {
    const robots = buildRobots(SITE);

    expect(robots).toContain("Sitemap: https://exemplo.test/sitemap.xml");
    expect(robots).toContain("Disallow: /api/");
    expect(robots).toContain("User-agent: *");
  });
});
