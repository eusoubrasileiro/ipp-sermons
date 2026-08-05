import type { SearchFilters } from "@ipp/shared";

/**
 * The query-string vocabulary the browse routes share.
 *
 * `/api/sermons` and `/api/facets/counts` take exactly the same filters, and
 * the frontend puts the same names in the address bar, so a shared sermon URL
 * and the request it produces read alike. One parser keeps the three in step.
 *
 * Lists are comma-separated (`?livros=efesios,genesis`); an absent key stays
 * `undefined` rather than becoming an empty array, because empty means "match
 * nothing" everywhere downstream.
 */
const asArray = (v: string | undefined): string[] | undefined =>
  v ? v.split(",").filter(Boolean) : undefined;

export function filtersFromQuery(q: Record<string, string>): Record<string, unknown> {
  return {
    pregadores: asArray(q.pregadores),
    tipos: asArray(q.tipos),
    series: asArray(q.series),
    livros: asArray(q.livros),
    temas: asArray(q.temas),
    capitulo: q.capitulo ? Number.parseInt(q.capitulo, 10) : undefined,
    de: q.de,
    ate: q.ate,
  } satisfies Record<keyof SearchFilters, unknown>;
}
