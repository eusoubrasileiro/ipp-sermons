import type { SearchFilters } from "@ipp/shared";
import { type Chip, chipsOf } from "../lib/facet-labels.ts";
import { useFacets } from "../lib/useFacets.ts";

/**
 * The active filters, one removable chip each.
 *
 * The whole chip is the remove button rather than a small ✕ inside it: a 24px
 * cross fails the 44px touch target the rest of the site keeps, and a chip has
 * no second action to compete with.
 *
 * The dimension is shown ("Bíblia: Efésios 5") because a bare "Efésios 5"
 * beside a preacher's name reads as a list of unrelated words.
 *
 * Rendered only when something is filtered, which is also what keeps the index
 * tree off the wire for the many visits that only ever search.
 */
export function FacetChips({
  filtros,
  onRemove,
}: {
  filtros: SearchFilters;
  onRemove: (chip: Chip) => void;
}) {
  const { facets } = useFacets();

  return (
    <>
      {chipsOf(filtros, facets).map((chip) => (
        <button
          key={`${chip.key}:${chip.value}`}
          type="button"
          onClick={() => onRemove(chip)}
          aria-label={`Remover filtro ${chip.label}`}
          className="inline-flex min-h-11 items-center gap-1 rounded-full border border-border bg-accent px-3 text-xs text-accent-foreground transition hover:border-destructive/50 hover:bg-destructive/10"
        >
          <span className="text-muted-foreground">{chip.dimensao}:</span>
          <span className="font-medium">{chip.label}</span>
          <span aria-hidden="true" className="pl-0.5 text-muted-foreground">
            ✕
          </span>
        </button>
      ))}
    </>
  );
}
