import type { SearchResult } from "@ipp/shared";
import { useState } from "react";
import { searchSermons } from "./api.ts";
import { SearchForm } from "./components/SearchForm.tsx";
import { SermonCard } from "./components/SermonCard.tsx";
import { EmptyState, ErrorState, IntroState, ResultsSkeleton } from "./components/States.tsx";
import { SuggestionBox } from "./components/SuggestionBox.tsx";
import { groupBySermon } from "./lib/group.ts";

/**
 * Single-page sermon search for Igreja Presbiteriana Peregrinos.
 *
 * Replaces a 5 KB static page, so it stays deliberately small: a search box,
 * result cards, and the feedback box the old server had. No router, no state
 * library, no component framework beyond Tailwind tokens.
 */

type Status = "idle" | "loading" | "done" | "error";

export function App() {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [tookMs, setTookMs] = useState(0);

  async function runSearch(q: string): Promise<void> {
    const trimmed = q.trim();
    if (trimmed.length < 2) return;

    setStatus("loading");
    setError("");
    setSubmitted(trimmed);

    try {
      const res = await searchSermons(trimmed);
      setResults(res.results);
      setTookMs(res.tookMs);
      setStatus("done");
    } catch (err) {
      setResults([]);
      setError(err instanceof Error ? err.message : "A busca falhou.");
      setStatus("error");
    }
  }

  const groups = groupBySermon(results);

  return (
    <div className="min-h-dvh">
      <div className="mx-auto max-w-3xl px-4 pb-12">
        <header className="pt-6 sm:pt-10">
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Sermões IPP</h1>
          <p className="mt-1 text-sm text-muted-foreground sm:text-base">
            Busca nos sermões da Igreja Presbiteriana Peregrinos.
          </p>
        </header>

        <SearchForm
          query={query}
          onQueryChange={setQuery}
          onSearch={(q) => void runSearch(q)}
          loading={status === "loading"}
          showExamples={status !== "done" || groups.length === 0}
        />

        {/* Announced to screen readers so a search that returns nothing is not silent. */}
        <p aria-live="polite" className="sr-only">
          {status === "loading" ? "Buscando…" : ""}
          {status === "done"
            ? `${groups.length} ${groups.length === 1 ? "sermão encontrado" : "sermões encontrados"}`
            : ""}
        </p>

        <main className="mt-4">
          {status === "idle" && <IntroState />}
          {status === "loading" && <ResultsSkeleton />}
          {status === "error" && (
            <ErrorState message={error} onRetry={() => void runSearch(submitted || query)} />
          )}
          {status === "done" && groups.length === 0 && <EmptyState query={submitted} />}

          {status === "done" && groups.length > 0 && (
            <>
              <p className="mb-3 text-xs text-muted-foreground">
                {groups.length} {groups.length === 1 ? "sermão" : "sermões"} · {tookMs} ms
              </p>
              <div className="space-y-3">
                {groups.map((g) => (
                  <SermonCard key={g.top.id} group={g} query={submitted} />
                ))}
              </div>
            </>
          )}
        </main>

        <SuggestionBox />
      </div>
    </div>
  );
}
