import { useParams } from "react-router-dom";
import { type FacetTree, toBrowseQuery } from "../api.ts";
import type { FacetGroup } from "../components/FacetIndex.tsx";
import { FacetIndexPage } from "../components/FacetIndexPage.tsx";
import { useFacets } from "../lib/useFacets.ts";
import { BrowseResults } from "./BrowseResults.tsx";

/**
 * The series index and the page for one course.
 *
 * A series needs two parts to be a series; a one-off lesson is filed under
 * "Avulsos" rather than padding the index with 21 entries of one. 38% of the
 * corpus is Sunday school taught as multi-week courses, so "give me all
 * seventeen parts in order" is a real request this page exists to answer.
 */
function toGroups(facets: FacetTree): FacetGroup[] {
  const real = facets.series.filter((s) => s.total >= 2);
  const byLetter = new Map<string, FacetGroup>();

  // Numeric collation, or "CFW 23" sorts between "CFW 2" and "CFW 3" and the
  // Westminster chapters read out of order.
  const byName = (a: { nome: string }, b: { nome: string }) =>
    a.nome.localeCompare(b.nome, "pt-BR", { numeric: true });

  for (const s of [...real].sort(byName)) {
    const letter = s.nome.normalize("NFD").replace(/[̀-ͯ]/g, "")[0]?.toUpperCase() ?? "#";
    const group = byLetter.get(letter) ?? { label: letter, entries: [] };
    const parent = s.paiSlug ? facets.series.find((p) => p.slug === s.paiSlug) : undefined;
    group.entries.push({
      to: `/series/${s.slug}`,
      label: s.nome,
      detail: parent?.nome ?? s.descricao ?? undefined,
      total: s.total,
    });
    byLetter.set(letter, group);
  }

  const avulsos = facets.series.filter((s) => s.total === 1);
  const groups = [...byLetter.values()];
  if (avulsos.length > 0) {
    groups.push({
      label: "Avulsos",
      entries: [...avulsos]
        .sort(byName)
        .map((s) => ({ to: `/series/${s.slug}`, label: s.nome, total: s.total })),
    });
  }
  return groups;
}

export function SeriesPage() {
  const { facets } = useFacets();
  const { slug } = useParams();

  if (slug) {
    const serie = facets?.series.find((s) => s.slug === slug);
    return (
      <BrowseResults
        titulo={serie?.nome ?? slug}
        subtitulo={serie?.descricao || undefined}
        query={toBrowseQuery({ series: slug, ordenar: "serie" })}
        voltar={{ to: "/series", label: "Séries" }}
      />
    );
  }

  return <FacetIndexPage agrupar={toGroups} />;
}
