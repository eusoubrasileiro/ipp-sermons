import { describe, expect, it } from "vitest";
import { escapeHtml, renderPage, type SeoPage, summarise } from "../../src/lib/seo/html.ts";

/**
 * The injection itself.
 *
 * Everything that lands in these pages comes from `data/metadata.csv` and
 * `data/transcripts/` — a CSV no schema validates and 3.6 million words of
 * transcribed speech. Escaping is therefore not a nicety, and neither is the
 * refusal to inject into a shell that is not the one we build against: a
 * half-rewritten `index.html` served to a human is a broken site, and falling
 * through to the untouched shell is always the safe answer.
 */

const SHELL = `<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="UTF-8" />
    <title>Sermões IPP — Igreja Presbiteriana Peregrinos</title>
    <meta
      name="description"
      content="Busca nos sermões da Igreja Presbiteriana Peregrinos."
    />
    <script type="module" crossorigin src="/assets/index-abc.js"></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>`;

const page = (over: Partial<SeoPage> = {}): SeoPage => ({
  title: "Tito 2",
  description: "Um sermão sobre a graça.",
  path: "/sermao/123",
  ogType: "article",
  body: "<h1>Tito 2</h1>",
  ...over,
});

describe("escapeHtml", () => {
  it("neutralises every character that can close a tag or an attribute", () => {
    expect(escapeHtml(`<script>alert("x" & 'y')</script>`)).toBe(
      "&lt;script&gt;alert(&quot;x&quot; &amp; &#39;y&#39;)&lt;/script&gt;",
    );
  });

  it("escapes the ampersand before anything else, so entities are not doubled up", () => {
    expect(escapeHtml("Tiago & <b>")).toBe("Tiago &amp; &lt;b&gt;");
  });

  it("leaves ordinary Portuguese alone", () => {
    expect(escapeHtml("Gênesis 17.9-27 — a aliança")).toBe("Gênesis 17.9-27 — a aliança");
  });
});

describe("summarise", () => {
  it("collapses the whitespace a transcript carries", () => {
    expect(summarise("Como  sempre,\n é um motivo")).toBe("Como sempre, é um motivo");
  });

  it("cuts at a word boundary and marks the cut", () => {
    const long = `${"palavra ".repeat(40)}fim`;
    const got = summarise(long, 40);

    expect(got.length).toBeLessThanOrEqual(41);
    expect(got.endsWith("…")).toBe(true);
    expect(got).not.toContain("palavr…");
  });

  it("returns a short text untouched", () => {
    expect(summarise("Um sermão curto.")).toBe("Um sermão curto.");
  });
});

describe("renderPage", () => {
  it("replaces the shell's title and description with the page's own", () => {
    const html = renderPage(SHELL, page(), "https://exemplo.test") ?? "";

    expect(html).toContain("<title>Tito 2</title>");
    expect(html).toContain('<meta name="description" content="Um sermão sobre a graça." />');
    // The shell's boilerplate must be gone, not merely outranked: two <title>
    // tags is undefined behaviour and two descriptions is a Search Console warning.
    expect(html).not.toContain("Igreja Presbiteriana Peregrinos</title>");
    expect(html.match(/<meta name="description"/g)).toHaveLength(1);
  });

  it("keeps the script tag that boots the SPA", () => {
    const html = renderPage(SHELL, page(), "https://exemplo.test") ?? "";
    expect(html).toContain('<script type="module" crossorigin src="/assets/index-abc.js">');
  });

  it("puts the body inside #root, where React will replace it on mount", () => {
    const html = renderPage(SHELL, page(), "https://exemplo.test") ?? "";
    expect(html).toContain('<div id="root"><h1>Tito 2</h1></div>');
  });

  it("writes an absolute canonical and the Open Graph tags a WhatsApp preview reads", () => {
    const html = renderPage(SHELL, page(), "https://exemplo.test") ?? "";

    expect(html).toContain('<link rel="canonical" href="https://exemplo.test/sermao/123" />');
    expect(html).toContain('<meta property="og:url" content="https://exemplo.test/sermao/123" />');
    expect(html).toContain('<meta property="og:title" content="Tito 2" />');
    expect(html).toContain('<meta property="og:type" content="article" />');
  });

  it("escapes a title and description before they reach an attribute", () => {
    const html =
      renderPage(
        SHELL,
        page({ title: `"><script>alert(1)</script>`, description: `Tiago & "aspas"` }),
        "https://exemplo.test",
      ) ?? "";

    expect(html).not.toContain("<script>alert(1)</script>");
    expect(html).toContain("&lt;script&gt;alert(1)&lt;/script&gt;");
    expect(html).toContain('content="Tiago &amp; &quot;aspas&quot;"');
  });

  it("does not let a dollar sign in a title be read as a capture reference", () => {
    // String.replace expands $&, $` and $' in a replacement STRING. A sermon
    // titled with one would splice the surrounding shell into the page.
    const html = renderPage(SHELL, page({ title: "Salmo $& $` 23" }), "https://exemplo.test") ?? "";
    expect(html).toContain("<title>Salmo $&amp; $` 23</title>");
  });

  it("refuses a shell with no #root rather than serving a half-written page", () => {
    expect(renderPage("<html><head></head><body></body></html>", page(), "https://x.test")).toBe(
      null,
    );
  });

  it("refuses a shell with no head", () => {
    expect(
      renderPage(`<html><body><div id="root"></div></body></html>`, page(), "https://x.test"),
    ).toBe(null);
  });
});
