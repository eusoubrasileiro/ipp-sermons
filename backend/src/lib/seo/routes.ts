import type { PrismaClient } from "@prisma/client";
import type { Context, Hono, Next } from "hono";
import { readSermonTranscript } from "../transcript.ts";
import { FAMILIES, facetIndexPage, facetLeafPage, homePage } from "./browse-pages.ts";
import { renderPage, type SeoPage } from "./html.ts";
import { sermonPage } from "./sermon-page.ts";
import type { ShellLoader } from "./shell.ts";
import { buildRobots, buildSitemap } from "./sitemap.ts";

/**
 * The crawlable half of the site.
 *
 * Registered inside `createApp()` and therefore ahead of the static middleware
 * and the SPA catch-all in `server.ts` — the same ordering trap the transcript
 * route documents. Behind them, every one of these would be answered with the
 * empty shell instead.
 *
 * The governing rule is that this may never make the site worse than it is
 * without it. A missing sermon, a transcript that is not on disk, a database
 * that is down, a facet slug nobody recognises, a frontend that was never
 * built: every one of them calls `next()` and lets the SPA answer exactly as it
 * does today. Nothing here returns an error page, because a search engine
 * seeing a 500 is worse than a search engine seeing what it already sees.
 */

type SeoDeps = {
  prisma: PrismaClient;
  /** Where `transcripts/` lives — `DATA_DIR` in production. */
  dataDir: string;
  shell: ShellLoader;
  /** Absolute origin for canonical and Open Graph URLs. */
  siteUrl: string;
};

/**
 * The prerendered HTML must NOT be held by a browser.
 *
 * It names the bundle in `<script src="/assets/index-<hash>.js">`, and a
 * release changes that hash. A held copy therefore asks a new container for a
 * file the old image had; the SPA catch-all answers with `index.html`, the
 * browser refuses it as a module (wrong MIME type) and the page never hydrates
 * — a blank site for whatever the max-age was. This was not hypothetical: it
 * happened during development the first time this shipped with `max-age=3600`.
 *
 * Revalidating is cheap. Rebuilding a page is one indexed query plus, for a
 * sermon, one 40 KB file read.
 */
const HTML_CACHE_CONTROL = "public, max-age=0, must-revalidate";

/** Neither of these names a hashed asset, so an hour costs nothing. */
const TEXT_CACHE_CONTROL = "public, max-age=3600";

/** Builds a page, or gives up and lets the SPA answer. */
function prerender(deps: SeoDeps, build: (c: Context) => Promise<SeoPage | null>) {
  return async (c: Context, next: Next) => {
    let page: SeoPage | null = null;
    try {
      page = await build(c);
    } catch (err) {
      console.error("[seo] prerender failed", err);
    }
    if (!page) return next();

    const shell = await deps.shell();
    const html = shell ? renderPage(shell, page, deps.siteUrl) : null;
    if (!html) return next();

    c.header("Cache-Control", HTML_CACHE_CONTROL);
    return c.html(html);
  };
}

export function registerSeoRoutes(app: Hono, deps: SeoDeps): void {
  app.get("/robots.txt", (c) => {
    c.header("Cache-Control", TEXT_CACHE_CONTROL);
    return c.text(buildRobots(deps.siteUrl));
  });

  app.get("/sitemap.xml", async (c) => {
    try {
      const xml = await buildSitemap(deps.prisma, deps.siteUrl);
      c.header("Cache-Control", TEXT_CACHE_CONTROL);
      return c.body(xml, 200, { "Content-Type": "application/xml; charset=UTF-8" });
    } catch (err) {
      console.error("[seo] sitemap failed", err);
      // 503 rather than an empty urlset: a sitemap that lists nothing is a
      // crawler's instruction to forget the URLs it already knows.
      return c.text("sitemap indisponível", 503);
    }
  });

  app.get(
    "/sermao/:id",
    prerender(deps, async (c) => {
      const transcript = await readSermonTranscript(
        deps.prisma,
        deps.dataDir,
        c.req.param("id") ?? "",
      );
      return transcript ? sermonPage(transcript) : null;
    }),
  );

  app.get(
    "/",
    prerender(deps, () => homePage(deps.prisma)),
  );

  for (const family of FAMILIES) {
    app.get(
      `/${family}`,
      prerender(deps, () => facetIndexPage(deps.prisma, family)),
    );
    app.get(
      `/${family}/:slug`,
      prerender(deps, (c) => facetLeafPage(deps.prisma, family, c.req.param("slug") ?? "")),
    );
  }

  // Only two facets go three deep -- a chapter and a month. The others would
  // resolve to nothing, and registering them would cost a facet-tree query to
  // discover that.
  for (const family of ["biblia", "datas"] as const) {
    app.get(
      `/${family}/:slug/:sub`,
      prerender(deps, (c) =>
        facetLeafPage(deps.prisma, family, c.req.param("slug") ?? "", c.req.param("sub")),
      ),
    );
  }
}
