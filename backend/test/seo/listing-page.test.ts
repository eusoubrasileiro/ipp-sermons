import { describe, expect, it } from "vitest";
import { listingPage } from "../../src/lib/seo/listing-page.ts";

/**
 * A browse page as a document.
 *
 * These are the crawl path: without real `<a href>` here, every one of the
 * 560 sermon URLs is an island reachable only through `sitemap.xml`.
 */

describe("listingPage", () => {
  const sermons = [
    {
      id: "1",
      title: "18-07-2021 - Gênesis 17.9-27",
      artist: "Reverendo Bruno Melo",
      date: "2021-07-18",
    },
    { id: "2", title: "Efésios 5", artist: "Pastor Lucas Antunes", date: "2022-02-06" },
  ];

  it("links every sermon it lists, which is how a crawler reaches them", () => {
    const page = listingPage({
      heading: "Gênesis",
      title: "Gênesis",
      description: "Sermões em Gênesis.",
      path: "/biblia/genesis",
      sermons,
      links: [],
      total: 2,
    });

    expect(page.body).toContain('href="/sermao/1"');
    expect(page.body).toContain('href="/sermao/2"');
    expect(page.body).toContain("Gênesis 17.9-27");
    expect(page.ogType).toBe("website");
  });

  it("links the facet index entries, so the tree is crawlable without JavaScript", () => {
    const page = listingPage({
      heading: "Bíblia",
      title: "Bíblia",
      description: "Índice.",
      path: "/biblia",
      sermons: [],
      links: [{ href: "/biblia/genesis", label: "Gênesis", total: 12 }],
    });

    expect(page.body).toContain('href="/biblia/genesis"');
    expect(page.body).toContain("Gênesis");
    expect(page.body).toContain("12");
  });

  it("escapes a facet label that carries markup", () => {
    const page = listingPage({
      heading: "<b>x</b>",
      title: "x",
      description: "d",
      path: "/temas",
      sermons: [],
      links: [{ href: "/temas/a", label: `<img src=x onerror=alert(1)>`, total: 1 }],
    });

    expect(page.body).not.toContain("<img src=x");
    expect(page.body).toContain("&lt;img src=x");
    expect(page.body).not.toContain("<b>x</b>");
  });

  it("says so when a facet has nothing under it rather than rendering a bare heading", () => {
    const page = listingPage({
      heading: "Temas",
      title: "Temas",
      description: "d",
      path: "/temas",
      sermons: [],
      links: [],
    });

    expect(page.body).toContain("Nenhum sermão");
  });
});
