import type { SearchFilters } from "@ipp/shared";
import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { FacetChips } from "../components/FacetChips.tsx";
import { FilterPicker } from "../components/FilterPicker.tsx";
import { SearchForm } from "../components/SearchForm.tsx";
import { SermonCard } from "../components/SermonCard.tsx";
import { EmptyState, ErrorState, IntroState, ResultsSkeleton } from "../components/States.tsx";
import { SuggestionBox } from "../components/SuggestionBox.tsx";
import type { Chip, DimensionKey } from "../lib/facet-labels.ts";
import {
  addFilter,
  countFilters,
  dropFilter,
  dropYear,
  parseFilters,
  toSearchParams,
  withYear,
} from "../lib/facet-params.ts";
import { groupBySermon } from "../lib/group.ts";
import { useSearch } from "../lib/useSearch.ts";
import { BrowseResults } from "./BrowseResults.tsx";

/**
 * The search page: still the hero of the site, now with filters that compose
 * with the query rather than replacing it.
 *
 * Everything lives in the address bar -- query and chips both -- so a narrowed
 * search survives a reload and can be forwarded to someone in WhatsApp. That
 * is also why there is no filter state in this component: the URL is it.
 */
export function SearchPage() {
  const [params, setParams] = useSearchParams();
  const query = params.get("q") ?? "";
  const filtros = parseFilters(params);
  const filterQuery = toSearchParams("", filtros).toString();

  // The box is a draft until submitted; the URL only moves on a real search.
  const [draft, setDraft] = useState(query);
  useEffect(() => setDraft(query), [query]);

  const { status, results, tookMs, error, retry } = useSearch(query, filterQuery);
  const groups = groupBySermon(results);
  const filtrado = countFilters(filtros) > 0;

  const apply = (q: string, next: SearchFilters): void => setParams(toSearchParams(q, next));

  const add = (key: DimensionKey, value: string): void =>
    apply(query, key === "ano" ? withYear(filtros, Number(value)) : addFilter(filtros, key, value));

  const remove = (chip: Chip): void =>
    apply(
      query,
      chip.key === "ano" ? dropYear(filtros) : dropFilter(filtros, chip.key, chip.value),
    );

  // Chips with nothing typed are still a request: show what they select rather
  // than sit on the intro screen. The search route cannot serve it -- it needs
  // two characters of query -- so this is the plain listing.
  const listando = query.trim().length < 2 && filtrado;

  return (
    <>
      <SearchForm
        query={draft}
        onQueryChange={setDraft}
        onSearch={(q) => apply(q, filtros)}
        loading={status === "loading"}
        showExamples={!filtrado && (status !== "done" || groups.length === 0)}
      />

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        {filtrado ? <FacetChips filtros={filtros} onRemove={remove} /> : null}
        <FilterPicker filtros={filtros} onAdd={add} />
      </div>

      {/* Announced to screen readers so a search that returns nothing is not silent. */}
      <p aria-live="polite" className="sr-only">
        {status === "loading" ? "Buscando…" : ""}
        {status === "done"
          ? `${groups.length} ${groups.length === 1 ? "sermão encontrado" : "sermões encontrados"}`
          : ""}
      </p>

      <main className="mt-4">
        {listando ? <BrowseResults titulo="Sermões filtrados" query={filterQuery} /> : null}

        {!listando && status === "idle" && <IntroState />}
        {status === "loading" && <ResultsSkeleton />}
        {status === "error" && <ErrorState message={error} onRetry={retry} />}

        {status === "done" && groups.length === 0 && (
          <EmptyState
            query={query}
            acao={
              filtrado ? { label: "Limpar os filtros", onClick: () => apply(query, {}) } : undefined
            }
          />
        )}

        {status === "done" && groups.length > 0 && (
          <>
            <p className="mb-3 text-xs text-muted-foreground">
              {groups.length} {groups.length === 1 ? "sermão" : "sermões"} · {tookMs} ms
            </p>
            <div className="space-y-3">
              {groups.map((g) => (
                <SermonCard key={g.top.id} group={g} query={query} />
              ))}
            </div>
          </>
        )}
      </main>

      <SuggestionBox />
    </>
  );
}
