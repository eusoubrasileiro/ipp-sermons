import { useEffect, useState } from "react";
import { type FacetTree, fetchFacets } from "../api.ts";

/**
 * The index tree, fetched once and shared by every browse page.
 *
 * Module-level cache rather than a state library: the tree is a few kilobytes
 * and only changes on re-index, so refetching it when someone taps between
 * Bíblia and Séries would be pure latency.
 */
let cached: FacetTree | null = null;
let inFlight: Promise<FacetTree> | null = null;

/**
 * Drops the cached tree.
 *
 * The cache is deliberately module-level and lives for the whole session, so
 * tests share it unless they say otherwise -- and a test that cannot see a
 * failed load is not testing the failure.
 */
export function clearFacetsCache(): void {
  cached = null;
  inFlight = null;
}

export function useFacets(): { facets: FacetTree | null; error: string } {
  const [facets, setFacets] = useState<FacetTree | null>(cached);
  const [error, setError] = useState("");

  useEffect(() => {
    if (cached) return;
    let alive = true;

    inFlight ??= fetchFacets();
    inFlight
      .then((tree) => {
        cached = tree;
        if (alive) setFacets(tree);
      })
      .catch((err: unknown) => {
        inFlight = null;
        if (alive) setError(err instanceof Error ? err.message : "Falha ao carregar.");
      });

    return () => {
      alive = false;
    };
  }, []);

  return { facets, error };
}
