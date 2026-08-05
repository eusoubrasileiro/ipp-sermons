import { type RenderResult, render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, vi } from "vitest";
import { App } from "./App.tsx";
import { clearFacetsCache } from "./lib/useFacets.ts";
import { okResponse } from "./test-fixtures.tsx";

/**
 * The browse side: the five index pages and the listings they link to.
 *
 * Driven through the real router at a real URL, because addressability is the
 * feature -- a member sends /biblia/efesios/5 to someone and it has to open
 * there.
 */
export const FACETS = {
  livros: [
    {
      slug: "genesis",
      nome: "Gênesis",
      testamento: "AT",
      ordem: 1,
      total: 73,
      capitulos: [
        { numero: 1, total: 6 },
        { numero: 3, total: 4 },
      ],
    },
    {
      slug: "efesios",
      nome: "Efésios",
      testamento: "NT",
      ordem: 49,
      total: 64,
      capitulos: [{ numero: 5, total: 13 }],
    },
  ],
  series: [
    { slug: "cfw-2", nome: "CFW 2 — De Deus", kind: "cfw", paiSlug: null, descricao: "", total: 4 },
    {
      slug: "cfw-23",
      nome: "CFW 23 — Magistrado",
      kind: "cfw",
      paiSlug: null,
      descricao: "",
      total: 4,
    },
    {
      slug: "cfw-3",
      nome: "CFW 3 — Decreto",
      kind: "cfw",
      paiSlug: "confissao",
      descricao: "Sobre o decreto eterno.",
      total: 5,
    },
    {
      slug: "confissao",
      nome: "Confissão de Fé de Westminster",
      kind: "cfw",
      paiSlug: null,
      descricao: "",
      total: 2,
    },
    { slug: "avulsa", nome: "Apologética", kind: "ebd", paiSlug: null, descricao: "", total: 1 },
  ],
  pregadores: [
    { slug: "reverendo-bruno-melo", nome: "Bruno Melo", titulo: "Reverendo", total: 268 },
    { slug: "pastor-lucas", nome: "Lucas Antunes", titulo: "Pastor", total: 18 },
  ],
  datas: [{ ano: 2024, total: 88, meses: [{ mes: 3, total: 9 }] }],
  tipos: [{ slug: "culto", total: 257 }],
  temas: [
    {
      slug: "ansiedade",
      nome: "Ansiedade",
      grupoSlug: "vida-crista",
      grupoNome: "Vida Cristã",
      total: 4,
    },
  ],
};

const SERMON = {
  id: "1",
  title: "17-03-2024 - Efésios 5.22-33 - O casamento",
  displayTitle: "Efésios 5.22-33 - O casamento",
  artist: "Reverendo Bruno Melo",
  date: "2024-03-17T00:00:00.000Z",
  durationStr: "48:25",
  scSuffixUrl: "casamento",
  spSuffixUrl: "abc",
  serviceType: "culto",
  seriesPart: null,
  series: null,
  scriptures: [{ bookSlug: "efesios", chapter: 5, book: { name: "Efésios" } }],
};

export const routeTo = (path: string): RenderResult =>
  render(<App />, {
    wrapper: ({ children }) => <MemoryRouter initialEntries={[path]}>{children}</MemoryRouter>,
  });

function stubBrowseFetch(): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url.startsWith("/api/facets")) return okResponse(FACETS);
      if (url.startsWith("/api/sermons")) {
        return okResponse({ total: 13, sermons: [SERMON], pagina: 1 });
      }
      return okResponse({});
    }),
  );
}

beforeEach(() => {
  clearFacetsCache();
  stubBrowseFetch();
});

afterEach(() => {
  vi.unstubAllGlobals();
});
