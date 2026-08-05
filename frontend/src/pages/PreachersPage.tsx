import { useParams } from "react-router-dom";
import { type FacetTree, toBrowseQuery } from "../api.ts";
import type { FacetGroup } from "../components/FacetIndex.tsx";
import { FacetIndexPage } from "../components/FacetIndexPage.tsx";
import { useFacets } from "../lib/useFacets.ts";
import { BrowseResults } from "./BrowseResults.tsx";

/**
 * Preachers, grouped by their honorific.
 *
 * Desiring God puts a photograph next to each author; we have none, so the
 * grouping carries the identity instead and the count does the rest.
 */
const ORDER = ["Reverendo", "Pastor", "Presbítero", "Seminarista", "Diácono", "Missionário"];

function toGroups(facets: FacetTree): FacetGroup[] {
  const byTitle = new Map<string, FacetGroup>();

  for (const p of facets.pregadores) {
    const group = byTitle.get(p.titulo) ?? { label: p.titulo, entries: [] };
    group.entries.push({ to: `/pregadores/${p.slug}`, label: p.nome, total: p.total });
    byTitle.set(p.titulo, group);
  }

  return [...byTitle.values()].sort(
    (a, b) =>
      (ORDER.indexOf(a.label) + 1 || 99) - (ORDER.indexOf(b.label) + 1 || 99) ||
      a.label.localeCompare(b.label, "pt-BR"),
  );
}

export function PreachersPage() {
  const { facets } = useFacets();
  const { slug } = useParams();

  if (slug) {
    const preacher = facets?.pregadores.find((p) => p.slug === slug);
    return (
      <BrowseResults
        titulo={preacher ? `${preacher.titulo} ${preacher.nome}` : slug}
        query={toBrowseQuery({
          pregadores: preacher ? `${preacher.titulo} ${preacher.nome}` : slug,
        })}
        voltar={{ to: "/pregadores", label: "Pregadores" }}
      />
    );
  }

  return <FacetIndexPage agrupar={toGroups} />;
}
