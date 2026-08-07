/**
 * The corpus as the browse tests see it.
 *
 * Kept apart from the render helpers so a test that only needs a shape can
 * import it without pulling in the fetch stub and its lifecycle hooks.
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
    {
      slug: "reverendo-bruno-melo",
      artist: "Reverendo Bruno Melo",
      nome: "Bruno Melo",
      titulo: "Reverendo",
      total: 268,
    },
    {
      slug: "pastor-lucas",
      artist: "Pastor Lucas Antunes",
      nome: "Lucas Antunes",
      titulo: "Pastor",
      total: 18,
    },
  ],
  datas: [{ ano: 2024, total: 88, meses: [{ mes: 3, total: 9 }] }],
  tipos: [
    { slug: "culto", total: 257 },
    { slug: "ebd", total: 176 },
  ],
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

export const SERMON = {
  id: "1",
  title: "17-03-2024 - Efésios 5.22-33 - O casamento",
  displayTitle: "Efésios 5.22-33 - O casamento",
  artist: "Reverendo Bruno Melo",
  date: "2024-03-17T00:00:00.000Z",
  durationStr: "48:25",
  scSuffixUrl: "casamento",
  spSuffixUrl: "abc",
  spotifyAlive: true,
  serviceType: "culto",
  seriesPart: null,
  series: null,
  scriptures: [{ bookSlug: "efesios", chapter: 5, book: { name: "Efésios" } }],
};

/**
 * Counts as the popover sees them: narrowed to the filters already chosen, so
 * "Lucas Antunes" is absent here even though the tree above has him.
 */
export const COUNTS = {
  pregadores: { "Reverendo Bruno Melo": 2 },
  tipos: { culto: 2 },
  series: {},
  livros: { efesios: 2 },
  temas: {},
  anos: { "2024": 2 },
  total: 2,
};

export const SEARCH_RESULT = {
  id: "1",
  title: "17-03-2024 - Efésios 5.22-33 - O casamento",
  artist: "Reverendo Bruno Melo",
  date: "2024-03-17",
  durationStr: "48:25",
  soundcloudUrl: "https://soundcloud.com/ipperegrinos/casamento",
  spotifyUrl: null,
  content: "o casamento diante da cruz",
  score: 0.03,
  chunkIndex: 1,
};
