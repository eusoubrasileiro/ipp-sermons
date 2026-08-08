import { formatDate, stripLeadingDate } from "@ipp/shared";
import { escapeHtml, type SeoPage } from "./html.ts";
import { SHELL_CLOSE, SHELL_OPEN } from "./layout.ts";

/**
 * A browse page — an index of facets, or the sermons under one of them.
 *
 * These matter less than the sermons themselves and more than they look: they
 * are the only crawl path into the corpus that does not depend on the sitemap
 * being fetched and trusted. Without real `<a href>` here, every one of the 560
 * sermon URLs is an island.
 *
 * The listing is capped rather than paginated (see `browse-pages.ts`): a
 * crawler following `?pagina=2` links would pay for one query per page, and the
 * largest facet in this archive fits comfortably in one document.
 */

/** Only what a link needs — the listing never loads a transcript. */
export type ListedSermon = {
  id: string;
  title: string;
  artist: string;
  /** ISO, "2021-07-18". */
  date: string;
};

/** `total` is omitted where there is no meaningful count — the home page's index links. */
export type FacetLink = { href: string; label: string; total?: number | undefined };

type ListingInput = {
  /** The `<h1>`. */
  heading: string;
  title: string;
  description: string;
  path: string;
  sermons: ListedSermon[];
  links: FacetLink[];
  /** How many sermons the facet has, when more exist than are listed. */
  total?: number | undefined;
};

const linkList = (links: FacetLink[]): string =>
  links
    .map(
      (link) =>
        `<li><a class="text-primary underline-offset-4 hover:underline" href="${escapeHtml(
          link.href,
        )}">${escapeHtml(link.label)}</a>${
          link.total === undefined
            ? ""
            : ` <span class="text-muted-foreground">(${link.total})</span>`
        }</li>`,
    )
    .join("");

const sermonList = (sermons: ListedSermon[]): string =>
  sermons
    .map(
      (sermon) =>
        `<li><a class="text-primary underline-offset-4 hover:underline" href="/sermao/${escapeHtml(
          encodeURIComponent(sermon.id),
        )}">${escapeHtml(stripLeadingDate(sermon.title))}</a>` +
        ` <span class="text-muted-foreground">— ${escapeHtml(sermon.artist)}, ${escapeHtml(
          formatDate(sermon.date),
        )}</span></li>`,
    )
    .join("");

export function listingPage(input: ListingInput): SeoPage {
  const shown = input.sermons.length;
  const total = input.total ?? shown;

  const sections = [
    input.links.length > 0 ? `<ul class="mt-6 space-y-2">${linkList(input.links)}</ul>` : "",
    shown > 0 ? `<ul class="mt-6 space-y-2">${sermonList(input.sermons)}</ul>` : "",
    // An empty facet is a real state -- the topic pass has not run over every
    // sermon -- and a bare heading with nothing under it looks like a failure.
    shown === 0 && input.links.length === 0
      ? `<p class="mt-6 text-muted-foreground">Nenhum sermão aqui ainda.</p>`
      : "",
    total > shown
      ? `<p class="mt-4 text-sm text-muted-foreground">${total} sermões no total.</p>`
      : "",
  ];

  const body = [
    SHELL_OPEN,
    `<h1 class="font-display text-2xl font-bold leading-snug text-card-foreground sm:text-3xl">${escapeHtml(
      input.heading,
    )}</h1>`,
    `<p class="mt-2 text-sm text-muted-foreground">${escapeHtml(input.description)}</p>`,
    ...sections,
    SHELL_CLOSE,
  ].join("");

  return {
    title: input.title,
    description: input.description,
    path: input.path,
    ogType: "website",
    body,
  };
}
