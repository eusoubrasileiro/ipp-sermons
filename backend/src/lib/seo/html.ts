/**
 * Turning the built SPA shell into a page a search engine can read.
 *
 * `backend/public/index.html` answers every URL on this site, so to a crawler
 * 560 sermons and every browse page are one empty document with one title. This
 * module rewrites that shell per request: the page's real `<title>`, its own
 * description, a canonical URL, the Open Graph tags a WhatsApp preview reads,
 * and the page's text inside `#root`.
 *
 * It is string injection rather than `renderToString` on purpose. The runtime
 * image ships `backend/dist` and `backend/public` only — React and the JSX
 * pages are not in it, and pulling them in would mean bundling the frontend
 * into the server for a body that `createRoot()` throws away on mount anyway.
 * `createRoot` (not `hydrateRoot`) clears its container before the first
 * render, so nothing here has to match React's output to the byte; it only has
 * to carry the same words.
 *
 * Everything injected comes from `data/metadata.csv` and `data/transcripts/` —
 * a CSV no schema validates and 3.6 million words of transcribed speech — so
 * every interpolation goes through `escapeHtml`. Each `String.replace` takes a
 * function rather than a replacement string: `$&`, `` $` `` and `$'` are
 * expanded in a replacement string, and a sermon title containing one would
 * splice the surrounding shell into the page.
 */

const ESCAPES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

/** Safe for both text nodes and double-quoted attributes. */
export function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (ch) => ESCAPES[ch] ?? ch);
}

/** About what Google will render before truncating it itself. */
const DESCRIPTION_MAX = 160;

/**
 * A meta description cut from running text.
 *
 * Cuts at a word boundary and says so with an ellipsis: a description ending
 * mid-word reads as a broken page in the one place a stranger decides whether
 * to click.
 */
export function summarise(text: string, max: number = DESCRIPTION_MAX): string {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat.length <= max) return flat;

  const cut = flat.slice(0, max);
  const lastSpace = cut.lastIndexOf(" ");
  const body = lastSpace > max / 2 ? cut.slice(0, lastSpace) : cut;
  return `${body.replace(/[\s.,;:—–-]+$/u, "")}…`;
}

export type SeoPage = {
  /** Becomes `<title>` and `og:title`. */
  title: string;
  description: string;
  /** Site-absolute path, canonicalised — "/sermao/2123517330". */
  path: string;
  /** `article` for a sermon, `website` for an index or a listing. */
  ogType: "article" | "website";
  /** Markup for `#root`. Escaped by whoever built it. */
  body: string;
};

const TITLE_TAG = /<title>[^<]*<\/title>/i;
const DESCRIPTION_TAG = /<meta\s[^>]*name="description"[^>]*>/i;
const ROOT_DIV = /<div id="root">\s*<\/div>/;

const SITE_NAME = "Sermões IPP";

function headFor(page: SeoPage, siteUrl: string): string {
  const url = escapeHtml(`${siteUrl}${page.path}`);
  const title = escapeHtml(page.title);
  const description = escapeHtml(page.description);

  return [
    `<title>${title}</title>`,
    `<meta name="description" content="${description}" />`,
    `<link rel="canonical" href="${url}" />`,
    `<meta property="og:type" content="${page.ogType}" />`,
    `<meta property="og:title" content="${title}" />`,
    `<meta property="og:description" content="${description}" />`,
    `<meta property="og:url" content="${url}" />`,
    `<meta property="og:site_name" content="${SITE_NAME}" />`,
    `<meta property="og:locale" content="pt_BR" />`,
    `<meta name="twitter:card" content="summary" />`,
  ].join("\n    ");
}

/**
 * The shell with this page's head and body written into it.
 *
 * Returns null when the shell is not the document we build against — no
 * `#root` to fill, or no `</head>` to write into. The caller then serves the
 * shell untouched, which is exactly today's behaviour: a prerenderer that
 * half-rewrites a page is worse than one that declines.
 */
export function renderPage(shell: string, page: SeoPage, siteUrl: string): string | null {
  if (!ROOT_DIV.test(shell) || !shell.includes("</head>")) return null;

  const head = headFor(page, siteUrl);

  return shell
    .replace(TITLE_TAG, "")
    .replace(DESCRIPTION_TAG, "")
    .replace("</head>", () => `  ${head}\n  </head>`)
    .replace(ROOT_DIV, () => `<div id="root">${page.body}</div>`);
}
