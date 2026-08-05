import { useParams } from "react-router-dom";
import { type FacetTree, toBrowseQuery } from "../api.ts";
import type { FacetGroup } from "../components/FacetIndex.tsx";
import { FacetIndexPage } from "../components/FacetIndexPage.tsx";
import { useFacets } from "../lib/useFacets.ts";
import { BrowseResults } from "./BrowseResults.tsx";

/**
 * Two-level topic taxonomy: the group carries what Desiring God splits into a
 * separate "themes" page, the leaf is the topic. One structure is right for
 * 456 sermons; two would be ceremony.
 *
 * Empty until the labelling pass has run, which is why the empty state says so
 * rather than pretending the archive has no topics.
 */
function toGroups(facets: FacetTree): FacetGroup[] {
  const byGroup = new Map<string, FacetGroup>();

  for (const t of facets.temas) {
    if (t.total === 0) continue;
    const group = byGroup.get(t.grupoSlug) ?? { label: t.grupoNome, entries: [] };
    group.entries.push({ to: `/temas/${t.slug}`, label: t.nome, total: t.total });
    byGroup.set(t.grupoSlug, group);
  }

  return [...byGroup.values()];
}

export function TopicsPage() {
  const { facets } = useFacets();
  const { slug } = useParams();

  if (slug) {
    const tema = facets?.temas.find((t) => t.slug === slug);
    return (
      <BrowseResults
        titulo={tema?.nome ?? slug}
        subtitulo={tema?.grupoNome}
        query={toBrowseQuery({ temas: slug })}
        voltar={{ to: "/temas", label: "Temas" }}
      />
    );
  }

  return (
    <FacetIndexPage
      agrupar={toGroups}
      vazio={
        <p className="mt-6 rounded-lg border border-border bg-card p-4 text-sm text-muted-foreground">
          Os temas ainda estão sendo preparados. Enquanto isso, navegue por{" "}
          <a className="underline" href="/biblia">
            Bíblia
          </a>
          ,{" "}
          <a className="underline" href="/series">
            Séries
          </a>{" "}
          ou use a busca.
        </p>
      }
    />
  );
}
