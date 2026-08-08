/**
 * How a sermon's title and date are written. All output is pt-BR.
 *
 * Shared for the same reason `audio-urls` is: the server now renders a sermon's
 * heading into `index.html` before the browser sees it (`backend/src/lib/seo/`)
 * and React writes the same heading again a moment later. Two copies of these
 * rules would show up as the page visibly rewriting itself under the reader,
 * and as a `<title>` that disagrees with the `<h1>` under it.
 */

const MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"];

/**
 * Corpus titles carry the preaching date as a prefix ("18-07-2021 - Gênesis
 * 17.9-27", sometimes with slashes). Every surface shows the date in its own
 * field, so repeating it in the heading only costs line width on a phone — and
 * in a `<title>`, it costs the first words a search result shows.
 */
export function stripLeadingDate(title: string): string {
  return title.replace(/^\s*\d{1,2}[-/]?\d{2}[-/]?\d{2,4}\s*[-–—]\s*/, "").trim() || title.trim();
}

/** The `<title>` for anything that is not one sermon and not one facet. */
export const SITE_TITLE = "Sermões IPP — Igreja Presbiteriana Peregrinos";

/**
 * The `<title>` for a sermon.
 *
 * Shared because two things write it: the server, into the document a crawler
 * and a link preview read, and the SPA, into the browser tab after a
 * client-side navigation. A page whose tab disagrees with its own `<h1>` reads
 * as the wrong page.
 */
export function sermonTitle(title: string, artist: string): string {
  return `${stripLeadingDate(title)} — ${artist}`;
}

/** "2021-07-18" -> "18 jul 2021". Short enough to sit inline on a phone. */
export function formatDate(iso: string): string {
  const [year, month, day] = iso.split("-");
  const name = MONTHS[Number(month) - 1];
  if (!year || !day || !name) return iso;
  return `${day} ${name} ${year}`;
}
