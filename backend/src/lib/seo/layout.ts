/**
 * The wrapper the SPA puts around every page (`frontend/src/App.tsx`).
 *
 * Repeated here for two reasons. The obvious one is that the text painted
 * before the bundle arrives should sit at the width and margins the hydrated
 * page uses, and the tab bar should occupy its height, or the article visibly
 * drops down the screen when React takes over.
 *
 * The one that matters more: this is site-wide navigation in HTML. Every
 * prerendered page now links to every index, so a crawler that lands on one
 * sermon can reach the whole archive without executing JavaScript and without
 * having to trust `sitemap.xml`. It is the same six destinations as
 * `FacetNav`, which is the shortest list in this codebase and the least likely
 * to drift.
 *
 * The wordmark itself is deliberately not reproduced — it is an inline SVG
 * pilgrim in `components/Logo.tsx`, and a second copy of it here would be a
 * real maintenance cost for a logo nobody indexes.
 *
 * These utility classes are only in the stylesheet because `frontend/src` uses
 * them; Tailwind scans that tree, not this one. That is a cosmetic coupling
 * only — if a class disappears the prerendered text is briefly less pretty,
 * never absent or wrong.
 */

const TABS: [string, string][] = [
  ["/", "Buscar"],
  ["/temas", "Temas"],
  ["/biblia", "Bíblia"],
  ["/series", "Séries"],
  ["/pregadores", "Pregadores"],
  ["/datas", "Datas"],
];

const NAV = `<nav aria-label="Navegar no acervo" class="-mx-4 mt-3 overflow-x-auto border-b border-border px-4"><ul class="flex min-w-max gap-1 text-sm">${TABS.map(
  ([href, label]) =>
    `<li><a class="inline-flex min-h-11 items-center border-b-2 border-transparent px-3 text-muted-foreground hover:text-foreground" href="${href}">${label}</a></li>`,
).join("")}</ul></nav>`;

export const SHELL_OPEN =
  `<div class="min-h-dvh"><div class="mx-auto max-w-3xl px-4 pb-12">` +
  `<header class="pt-6 sm:pt-10"><a class="font-display text-2xl font-bold text-primary" href="/">Peregrinos</a>` +
  `<p class="mt-3 text-sm text-muted-foreground sm:text-base">Busque nos sermões por tema, passagem bíblica ou pregador.</p></header>` +
  `${NAV}<main class="mt-5">`;

export const SHELL_CLOSE = `</main></div></div>`;
