import type { SearchFilters } from "@ipp/shared";

/**
 * The filter state, serialised into the address bar.
 *
 * The URL is the single source of truth: reload, back button and a link
 * forwarded in WhatsApp all have to land on the same filtered search, and any
 * state kept beside the URL is a chance for the two to disagree.
 *
 * The parameter names are the API's own (`livros`, `pregadores`, …), so what
 * someone sees in the address bar is what the request carries. `pregadores`
 * holds the full name -- "Reverendo Bruno Melo" -- because that is what the
 * SQL filter compares against; a slug would need the facet tree loaded before
 * any filtered search could run.
 */
export type ListKey = "pregadores" | "tipos" | "series" | "livros" | "temas";

export const LIST_KEYS: ListKey[] = ["pregadores", "tipos", "series", "livros", "temas"];

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

const list = (sp: URLSearchParams, key: ListKey): string[] | undefined => {
  const values = (sp.get(key) ?? "").split(",").filter(Boolean);
  return values.length > 0 ? values : undefined;
};

const date = (sp: URLSearchParams, key: "de" | "ate"): string | undefined => {
  const value = sp.get(key) ?? "";
  return ISO_DATE.test(value) ? value : undefined;
};

export function parseFilters(sp: URLSearchParams): SearchFilters {
  const filters: SearchFilters = {};

  for (const key of LIST_KEYS) {
    const values = list(sp, key);
    if (values) filters[key] = values;
  }

  // A chapter only ever narrows a book, so on its own it is noise.
  const chapter = Number.parseInt(sp.get("capitulo") ?? "", 10);
  if (filters.livros && Number.isFinite(chapter)) filters.capitulo = chapter;

  const de = date(sp, "de");
  const ate = date(sp, "ate");
  if (de) filters.de = de;
  if (ate) filters.ate = ate;

  return filters;
}

export function toSearchParams(query: string, filters: SearchFilters): URLSearchParams {
  const sp = new URLSearchParams();
  if (query) sp.set("q", query);

  for (const key of LIST_KEYS) {
    const values = filters[key];
    if (values?.length) sp.set(key, values.join(","));
  }
  if (filters.livros?.length && filters.capitulo) sp.set("capitulo", String(filters.capitulo));
  if (filters.de) sp.set("de", filters.de);
  if (filters.ate) sp.set("ate", filters.ate);

  return sp;
}

export function addFilter(filters: SearchFilters, key: ListKey, value: string): SearchFilters {
  const current = filters[key] ?? [];
  if (current.includes(value)) return filters;
  return { ...filters, [key]: [...current, value] };
}

export function dropFilter(filters: SearchFilters, key: ListKey, value: string): SearchFilters {
  const next = { ...filters };
  const kept = (filters[key] ?? []).filter((v) => v !== value);

  if (kept.length > 0) next[key] = kept;
  else delete next[key];

  // The chapter belongs to the book: without one it filters nothing and would
  // still render as a chip.
  if (key === "livros" && kept.length === 0) delete next.capitulo;

  return next;
}

/** A year, as the date range the API takes. Replaces any range already set. */
export function withYear(filters: SearchFilters, year: number): SearchFilters {
  return { ...filters, de: `${year}-01-01`, ate: `${year}-12-31` };
}

export function dropYear(filters: SearchFilters): SearchFilters {
  const next = { ...filters };
  delete next.de;
  delete next.ate;
  return next;
}

/** One per chip on screen — the date range counts once, and so does a chapter. */
export function countFilters(filters: SearchFilters): number {
  const values = LIST_KEYS.reduce((n, key) => n + (filters[key]?.length ?? 0), 0);
  return values + (filters.de || filters.ate ? 1 : 0);
}
