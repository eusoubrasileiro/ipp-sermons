import type { FacetTree } from "../api.ts";
import { useFacets } from "../lib/useFacets.ts";
import { type FacetGroup, FacetIndex } from "./FacetIndex.tsx";

/**
 * Loading, error and empty handling shared by all five index pages.
 *
 * Each page differs only in how it groups the tree, so that grouping is the
 * one thing it passes in. `vazio` lets a page say something better than
 * "nothing here" when its emptiness is expected -- Temas has no entries until
 * the labelling pass has run.
 */
export function FacetIndexPage({
  agrupar,
  vazio,
}: {
  agrupar: (facets: FacetTree) => FacetGroup[];
  vazio?: React.ReactNode;
}) {
  const { facets, error } = useFacets();

  if (error) {
    return (
      <p role="alert" className="mt-6 text-sm text-destructive">
        {error}
      </p>
    );
  }
  if (!facets) {
    return (
      <p aria-live="polite" className="mt-6 text-sm text-muted-foreground">
        Carregando…
      </p>
    );
  }

  const groups = agrupar(facets);
  if (groups.length === 0 && vazio) return <>{vazio}</>;

  return <FacetIndex groups={groups} />;
}
