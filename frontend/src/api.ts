import type { SearchFilters, SearchResponse, TranscriptResponse } from "@ipp/shared";

/** Thin API client. Same origin in production; Vite proxies /api in dev. */

export async function searchSermons(
  query: string,
  filtros: SearchFilters = {},
  limit = 10,
): Promise<SearchResponse> {
  const res = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit, filtros }),
  });

  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error ?? "A busca falhou. Tente novamente.");
  }

  return (await res.json()) as SearchResponse;
}

export async function sendSuggestion(suggestion: string): Promise<void> {
  const res = await fetch("/api/suggestion", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ suggestion }),
  });
  if (!res.ok) throw new Error("Não foi possível enviar a sugestão.");
}

/** The whole browse index tree. Fetched once; every index page renders from it. */
export type FacetTree = {
  livros: {
    slug: string;
    nome: string;
    testamento: string;
    ordem: number;
    total: number;
    capitulos: { numero: number; total: number }[];
  }[];
  series: {
    slug: string;
    nome: string;
    kind: string;
    paiSlug: string | null;
    descricao: string;
    total: number;
  }[];
  /** `artist` is the raw name the filters compare against; `titulo` only groups. */
  pregadores: { slug: string; artist: string; nome: string; titulo: string; total: number }[];
  datas: { ano: number; total: number; meses: { mes: number; total: number }[] }[];
  tipos: { slug: string; total: number }[];
  temas: { slug: string; nome: string; grupoSlug: string; grupoNome: string; total: number }[];
};

export async function fetchFacets(): Promise<FacetTree> {
  const res = await fetch("/api/facets");
  if (!res.ok) throw new Error("Não foi possível carregar os índices.");
  return (await res.json()) as FacetTree;
}

/**
 * The same facets counted under the filters already chosen.
 *
 * What the "+ filtro" popover shows, so that no option on offer leads to an
 * empty page. Keyed by the value the filter uses, not by slug.
 */
export type FacetCounts = {
  pregadores: Record<string, number>;
  tipos: Record<string, number>;
  series: Record<string, number>;
  livros: Record<string, number>;
  temas: Record<string, number>;
  anos: Record<string, number>;
  total: number;
};

export async function fetchFacetCounts(query: string): Promise<FacetCounts> {
  const res = await fetch(`/api/facets/counts?${query}`);
  if (!res.ok) throw new Error("Não foi possível carregar os índices.");
  return (await res.json()) as FacetCounts;
}

export type BrowseSermon = {
  id: string;
  title: string;
  displayTitle: string | null;
  artist: string;
  date: string;
  durationStr: string;
  scSuffixUrl: string | null;
  spSuffixUrl: string | null;
  /** Whether the Spotify episode is still in the podcast feed, so still playable. */
  spotifyAlive: boolean;
  serviceType: string | null;
  seriesPart: number | null;
  series: { slug: string; name: string } | null;
  scriptures: { bookSlug: string; chapter: number; book: { name: string } }[];
};

/** Serialises a facet selection into the query string /api/sermons expects. */
export function toBrowseQuery(params: Record<string, string | number | undefined>): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") query.set(key, String(value));
  }
  return query.toString();
}

export async function browseSermons(
  query: string,
): Promise<{ total: number; sermons: BrowseSermon[]; pagina: number }> {
  const res = await fetch(`/api/sermons?${query}`);
  if (!res.ok) throw new Error("Não foi possível listar os sermões.");
  return (await res.json()) as { total: number; sermons: BrowseSermon[]; pagina: number };
}

/**
 * A whole sermon, for the reading page. Cached for a day by the server, so a
 * reader moving between results and transcript pays for it once.
 */
export async function fetchTranscript(id: string): Promise<TranscriptResponse> {
  const res = await fetch(`/api/sermons/${encodeURIComponent(id)}/transcript`);
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error ?? "Não foi possível carregar a transcrição.");
  }
  return (await res.json()) as TranscriptResponse;
}
