import { describe, expect, it } from "vitest";
import { SHELL_CLOSE, SHELL_OPEN } from "../../src/lib/seo/layout.ts";

/**
 * The wrapper every prerendered page sits inside.
 *
 * These are constants, so the tests are about the two things that make them
 * load-bearing rather than decorative: the markup has to close cleanly, or a
 * crawler parses the page wrong; and the nav has to link every index, because
 * it is the only way a crawler reaches the archive without JavaScript. A
 * sitemap alone is a request to be crawled, not a guarantee.
 */

const INDEXES = ["/", "/temas", "/biblia", "/series", "/pregadores", "/datas"];

describe("the prerendered shell", () => {
  it("opens and closes the same number of elements", () => {
    const html = `${SHELL_OPEN}<p>conteúdo</p>${SHELL_CLOSE}`;
    for (const tag of ["div", "main", "header", "nav", "ul"]) {
      const open = html.match(new RegExp(`<${tag}[\\s>]`, "g"))?.length ?? 0;
      const close = html.match(new RegExp(`</${tag}>`, "g"))?.length ?? 0;
      expect(`${tag}:${open}`).toBe(`${tag}:${close}`);
    }
  });

  it("links every browse index, so the corpus is reachable without JavaScript", () => {
    // This is the crawl path. Drop one and that whole facet becomes invisible
    // except to whatever the sitemap is trusted for.
    for (const href of INDEXES) {
      expect(SHELL_OPEN).toContain(`href="${href}"`);
    }
  });

  it("puts the archive behind a named landmark", () => {
    expect(SHELL_OPEN).toContain('aria-label="Navegar no acervo"');
    expect(SHELL_OPEN).toContain("<main");
  });

  it("leaves the article where the hydrated page puts it", () => {
    // Same width and gutters as frontend/src/App.tsx. If these drift the text
    // painted before the bundle lands jumps sideways when React takes over.
    expect(SHELL_OPEN).toContain("max-w-3xl");
    expect(SHELL_OPEN).toContain("px-4");
  });

  it("carries no unescaped interpolation", () => {
    // Everything here is a literal; a `${` surviving into the output would mean
    // a template was built wrong and is being served to every visitor.
    expect(SHELL_OPEN).not.toContain("${");
    expect(SHELL_CLOSE).not.toContain("${");
  });
});
