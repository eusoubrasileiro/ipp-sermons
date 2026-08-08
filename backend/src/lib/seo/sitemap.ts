import type { PrismaClient } from "@prisma/client";
import { facetTree } from "../browse/facets.ts";
import { FAMILIES } from "./browse-pages.ts";

/**
 * Every URL this site is willing to be found at.
 *
 * Generated from the database on request rather than committed. The corpus
 * grows by 50–100 sermons a year through `tools/corpus-update`, and a committed
 * sitemap is one more artifact that goes stale between releases without anyone
 * noticing — the failure mode being that the newest sermons are precisely the
 * ones nobody can find.
 *
 * One file: the protocol caps a sitemap at 50,000 URLs and 50 MB, and 560
 * sermons plus a few hundred facets is nowhere near either.
 */

const XML_ESCAPES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&apos;",
};

const escapeXml = (value: string): string =>
  value.replace(/[&<>"']/g, (ch) => XML_ESCAPES[ch] ?? ch);

type Entry = { path: string; lastmod?: string | undefined };

const urlTag = (siteUrl: string, entry: Entry): string =>
  `  <url>\n    <loc>${escapeXml(`${siteUrl}${entry.path}`)}</loc>${
    entry.lastmod ? `\n    <lastmod>${entry.lastmod}</lastmod>` : ""
  }\n  </url>`;

async function facetEntries(prisma: PrismaClient): Promise<Entry[]> {
  const tree = await facetTree(prisma);
  const entries: Entry[] = FAMILIES.map((family) => ({ path: `/${family}` }));

  for (const book of tree.livros) {
    entries.push({ path: `/biblia/${book.slug}` });
    for (const chapter of book.capitulos) {
      entries.push({ path: `/biblia/${book.slug}/${chapter.numero}` });
    }
  }
  for (const topic of tree.temas) {
    if (topic.total > 0) entries.push({ path: `/temas/${topic.slug}` });
  }
  for (const series of tree.series) {
    if (series.total > 0) entries.push({ path: `/series/${series.slug}` });
  }
  for (const preacher of tree.pregadores) {
    entries.push({ path: `/pregadores/${preacher.slug}` });
  }
  for (const year of tree.datas) {
    entries.push({ path: `/datas/${year.ano}` });
  }

  return entries;
}

export async function buildSitemap(prisma: PrismaClient, siteUrl: string): Promise<string> {
  const [facets, sermons] = await Promise.all([
    facetEntries(prisma),
    prisma.sermon.findMany({ select: { id: true, date: true }, orderBy: [{ date: "desc" }] }),
  ]);

  const entries: Entry[] = [
    { path: "/" },
    ...facets,
    ...sermons.map((sermon) => ({
      // Six sermons predate SoundCloud and fall back to their title as an id,
      // so an id is not guaranteed to be URL-safe.
      path: `/sermao/${encodeURIComponent(sermon.id)}`,
      // The preaching date, which is the only date this archive has. A sermon's
      // text changes only when it is re-transcribed, and that is rare enough
      // that claiming "modified today" on every crawl would be a lie.
      lastmod: sermon.date.toISOString().slice(0, 10),
    })),
  ];

  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...entries.map((entry) => urlTag(siteUrl, entry)),
    "</urlset>",
    "",
  ].join("\n");
}

/**
 * `Disallow: /api/` is not about secrecy — the API is public and read-only.
 * It keeps a crawler from spending this site's budget on JSON endpoints, one of
 * which (`POST /api/search`) costs a paid embedding call per request.
 */
export function buildRobots(siteUrl: string): string {
  return [
    "User-agent: *",
    "Allow: /",
    "Disallow: /api/",
    "",
    `Sitemap: ${siteUrl}/sitemap.xml`,
    "",
  ].join("\n");
}
