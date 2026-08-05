import { useEffect, useState } from "react";
import { type FacetCounts, fetchFacetCounts } from "../api.ts";

/**
 * Facet counts narrowed to the filters already chosen.
 *
 * Not cached like the index tree: these change with every chip, and a stale
 * count is worse than no count -- it offers a filter that empties the page.
 * Fetched only while the picker is open, so the search page still costs one
 * request for someone who never opens it.
 */
export function useFacetCounts(filterQuery: string, enabled: boolean): FacetCounts | null {
  const [counts, setCounts] = useState<FacetCounts | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let alive = true;

    fetchFacetCounts(filterQuery)
      .then((next) => {
        if (alive) setCounts(next);
      })
      .catch(() => {
        if (alive) setCounts(null);
      });

    return () => {
      alive = false;
    };
  }, [filterQuery, enabled]);

  return counts;
}
