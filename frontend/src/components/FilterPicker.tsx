import type { SearchFilters } from "@ipp/shared";
import { useState } from "react";
import { DIMENSIONS, type DimensionKey, optionsOf } from "../lib/facet-labels.ts";
import { toSearchParams } from "../lib/facet-params.ts";
import { useFacetCounts } from "../lib/useFacetCounts.ts";
import { useFacets } from "../lib/useFacets.ts";

/**
 * The "+ filtro" control.
 *
 * Expands in place instead of floating over the page: a true popover needs an
 * overlay, a focus trap and outside-click handling to be usable with a
 * keyboard, and on a phone it would cover the results it is narrowing.
 *
 * The counts come from /api/facets/counts, already narrowed to the chips
 * chosen, and anything at zero is not offered at all -- picking a filter must
 * never lead to an empty page.
 */
export function FilterPicker({
  filtros,
  onAdd,
}: {
  filtros: SearchFilters;
  onAdd: (key: DimensionKey, value: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-label="Adicionar filtro"
        className="inline-flex min-h-11 items-center rounded-full border border-dashed border-border px-3 text-xs text-muted-foreground transition hover:border-solid hover:bg-accent hover:text-accent-foreground"
      >
        + filtro
      </button>

      {/* Mounted only while open, so the search page costs one request for
          anyone who never touches the filters. */}
      {open ? (
        <FilterPanel
          filtros={filtros}
          onAdd={(key, value) => {
            onAdd(key, value);
            setOpen(false);
          }}
        />
      ) : null}
    </>
  );
}

/** Whether a value is already filtered on — those are not offered again. */
function isChosen(filtros: SearchFilters, key: DimensionKey, value: string): boolean {
  if (key === "ano") return filtros.de?.startsWith(value) === true;
  return (filtros[key] ?? []).includes(value);
}

function FilterPanel({
  filtros,
  onAdd,
}: {
  filtros: SearchFilters;
  onAdd: (key: DimensionKey, value: string) => void;
}) {
  const [dimensao, setDimensao] = useState<DimensionKey>("tipos");
  const { facets } = useFacets();
  const counts = useFacetCounts(toSearchParams("", filtros).toString(), true);

  const options = (facets && counts ? optionsOf(dimensao, facets, counts) : []).filter(
    (o) => o.total > 0 && !isChosen(filtros, dimensao, o.value),
  );

  return (
    <div className="mt-2 w-full rounded-lg border border-border bg-card p-3">
      <div className="flex flex-wrap gap-1">
        {DIMENSIONS.map((d) => (
          <button
            key={d.key}
            type="button"
            onClick={() => setDimensao(d.key)}
            aria-pressed={d.key === dimensao}
            className={[
              "min-h-9 rounded-md px-3 text-xs transition",
              d.key === dimensao
                ? "bg-primary font-medium text-primary-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
            ].join(" ")}
          >
            {d.label}
          </button>
        ))}
      </div>

      <div className="mt-2 max-h-60 overflow-y-auto border-t border-border pt-2">
        {!facets || !counts ? (
          <p className="px-1 py-2 text-xs text-muted-foreground">Carregando…</p>
        ) : null}

        {facets && counts && options.length === 0 ? (
          <p className="px-1 py-2 text-xs text-muted-foreground">
            Nada a acrescentar aqui com os filtros atuais.
          </p>
        ) : null}

        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            onClick={() => onAdd(dimensao, o.value)}
            // Read as one phrase, "Gênesis 73" becomes a chapter reference.
            aria-label={`${o.label}, ${o.total} ${o.total === 1 ? "sermão" : "sermões"}`}
            className="flex min-h-11 w-full items-center justify-between gap-3 rounded-md px-1 text-sm transition hover:bg-accent hover:text-accent-foreground"
          >
            <span className="min-w-0 truncate">{o.label}</span>
            <span
              aria-hidden="true"
              className="shrink-0 tabular-nums text-xs text-muted-foreground"
            >
              ({o.total})
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
