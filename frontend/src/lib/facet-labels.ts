import type { SearchFilters } from "@ipp/shared";
import type { FacetCounts, FacetTree } from "../api.ts";
import { LIST_KEYS, type ListKey } from "./facet-params.ts";

/**
 * Turns filter values into Portuguese the congregation reads.
 *
 * A filter travels as what the API compares against -- a book slug, a
 * preacher's full name, a year -- and none of those belong on screen. This is
 * the one place that maps between the two, so the chips and the "+ filtro"
 * popover can never disagree about what a filter is called.
 *
 * The order of the options is the facet tree's own: the backend already sorts
 * books canonically, series alphabetically and preachers by weight, and
 * re-sorting here would quietly contradict the index pages.
 */
const TYPE_LABELS: Record<string, string> = {
  culto: "Culto",
  ebd: "EBD",
  conferencia: "Conferência",
  congresso: "Congresso",
  confraria: "Confraria",
  diaconia: "Diaconia",
};

export type DimensionKey = ListKey | "ano";

type Option = { value: string; label: string; total: number };

const opt = (value: string, label: string, total: number | undefined): Option => ({
  value,
  label,
  total: total ?? 0,
});

const BUILDERS: Record<DimensionKey, (f: FacetTree, c: FacetCounts | null) => Option[]> = {
  tipos: (f, c) => f.tipos.map((t) => opt(t.slug, TYPE_LABELS[t.slug] ?? t.slug, c?.tipos[t.slug])),
  livros: (f, c) => f.livros.map((b) => opt(b.slug, b.nome, c?.livros[b.slug])),
  series: (f, c) => f.series.map((s) => opt(s.slug, s.nome, c?.series[s.slug])),
  // The full name is what the SQL filter matches; the first name is what reads.
  pregadores: (f, c) => f.pregadores.map((p) => opt(p.artist, p.nome, c?.pregadores[p.artist])),
  temas: (f, c) => f.temas.map((t) => opt(t.slug, t.nome, c?.temas[t.slug])),
  ano: (f, c) => f.datas.map((d) => opt(String(d.ano), String(d.ano), c?.anos[String(d.ano)])),
};

export const DIMENSIONS: { key: DimensionKey; label: string }[] = [
  { key: "tipos", label: "Tipo" },
  { key: "livros", label: "Bíblia" },
  { key: "series", label: "Série" },
  { key: "pregadores", label: "Pregador" },
  { key: "temas", label: "Tema" },
  { key: "ano", label: "Ano" },
];

const DIMENSION_LABEL = new Map(DIMENSIONS.map((d) => [d.key, d.label]));

export function optionsOf(
  key: DimensionKey,
  facets: FacetTree,
  counts: FacetCounts | null,
): Option[] {
  return BUILDERS[key](facets, counts);
}

const labelFor = (key: DimensionKey, value: string, facets: FacetTree | null): string =>
  (facets ? optionsOf(key, facets, null).find((o) => o.value === value)?.label : undefined) ??
  value;

/**
 * What a date range is called.
 *
 * The popover only ever sets whole years, but a shared link can carry any
 * range, so the odd case gets spelled out rather than mislabelled as a year.
 */
function rangeLabel(filters: SearchFilters): string | null {
  if (!filters.de && !filters.ate) return null;

  const de = filters.de?.slice(0, 4);
  if (de && de === filters.ate?.slice(0, 4)) return de;

  return [filters.de && `de ${filters.de}`, filters.ate && `até ${filters.ate}`]
    .filter(Boolean)
    .join(" ");
}

export type Chip = { key: DimensionKey; value: string; dimensao: string; label: string };

/** One chip per active filter — the chapter rides on its book, not on its own. */
export function chipsOf(filters: SearchFilters, facets: FacetTree | null): Chip[] {
  const chips: Chip[] = [];

  for (const key of LIST_KEYS) {
    for (const value of filters[key] ?? []) {
      const label = labelFor(key, value, facets);
      chips.push({
        key,
        value,
        dimensao: DIMENSION_LABEL.get(key) ?? key,
        label: key === "livros" && filters.capitulo ? `${label} ${filters.capitulo}` : label,
      });
    }
  }

  const range = rangeLabel(filters);
  if (range) chips.push({ key: "ano", value: range, dimensao: "Período", label: range });

  return chips;
}
