import type { SearchResponse } from "@ipp/shared";

/** Thin API client. Same origin in production; Vite proxies /api in dev. */

export async function searchSermons(query: string, limit = 10): Promise<SearchResponse> {
  const res = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit }),
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
