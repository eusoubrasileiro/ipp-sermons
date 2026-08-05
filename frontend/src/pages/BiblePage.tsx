import { useParams } from "react-router-dom";
import { type FacetTree, toBrowseQuery } from "../api.ts";
import type { FacetGroup } from "../components/FacetIndex.tsx";
import { FacetIndexPage } from "../components/FacetIndexPage.tsx";
import { useFacets } from "../lib/useFacets.ts";
import { BrowseResults } from "./BrowseResults.tsx";

/**
 * The scripture index, and the leaf page for a book or a chapter.
 *
 * Grouped by testament and ordered by the canon, never alphabetically: a Bible
 * that reads "Amós, Apocalipse, Atos, Cantares" is unusable to anyone who
 * knows it. Chapters hang under their book, and only where a sermon exists.
 */
function toGroups(facets: FacetTree): FacetGroup[] {
  const build = (testament: string) => ({
    label: testament === "AT" ? "Antigo Testamento" : "Novo Testamento",
    entries: facets.livros
      .filter((b) => b.testamento === testament)
      .map((b) => ({
        to: `/biblia/${b.slug}`,
        label: b.nome,
        total: b.total,
        children: b.capitulos
          .sort((x, y) => x.numero - y.numero)
          .map((c) => ({
            to: `/biblia/${b.slug}/${c.numero}`,
            label: `Capítulo ${c.numero}`,
            total: c.total,
          })),
      })),
  });

  return [build("AT"), build("NT")];
}

export function BiblePage() {
  const { facets } = useFacets();
  const { livro, capitulo } = useParams();

  if (livro) {
    const book = facets?.livros.find((b) => b.slug === livro);
    return (
      <BrowseResults
        titulo={book ? `${book.nome}${capitulo ? ` ${capitulo}` : ""}` : (livro ?? "")}
        query={toBrowseQuery({ livros: livro, capitulo })}
        voltar={{ to: "/biblia", label: "Bíblia" }}
      />
    );
  }

  return <FacetIndexPage agrupar={toGroups} />;
}
