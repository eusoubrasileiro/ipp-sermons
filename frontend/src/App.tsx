import type { SearchResult } from "@ipp/shared";
import { type FormEvent, useState } from "react";
import { searchSermons, sendSuggestion } from "./api.ts";
import { SermonCard } from "./SermonCard.tsx";

/**
 * Single-page sermon search for Igreja Presbiteriana Peregrinos.
 *
 * Replaces a 5 KB static page, so it stays deliberately small: a search box,
 * result cards, and the feedback box the old server had.
 */

const EXAMPLES = ["briga na igreja", "justificação pela fé", "batismo infantil", "Eclesiastes"];

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
      setError(err instanceof Error ? err.message : "A busca falhou.");
      setStatus("error");
    }
  }

  function onSubmit(e: FormEvent): void {
    e.preventDefault();
    void runSearch(query);
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100">
      <div className="mx-auto max-w-3xl px-4 py-10">
        <header className="mb-8">
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Sermões IPP</h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            Busca nos sermões da Igreja Presbiteriana Peregrinos — por tema, passagem ou pregador.
          </p>
        </header>

        <form onSubmit={onSubmit} className="mb-4">
          <label htmlFor="q" className="sr-only">
            O que você procura?
          </label>
          <div className="flex gap-2">
            <input
              id="q"
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="O que você procura?"
              className="w-full rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-base outline-none focus:border-slate-500 dark:border-slate-600 dark:bg-slate-800"
            />
            <button
              type="submit"
              disabled={status === "loading" || query.trim().length < 2}
              className="rounded-lg bg-slate-900 px-5 py-2.5 font-medium text-white disabled:opacity-40 dark:bg-slate-100 dark:text-slate-900"
            >
              {status === "loading" ? "Buscando…" : "Buscar"}
            </button>
          </div>
        </form>

        <div className="mb-8 flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-500 dark:text-slate-400">Exemplos:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => {
                setQuery(ex);
                void runSearch(ex);
              }}
              className="rounded-full border border-slate-300 px-3 py-1 text-xs text-slate-700 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              {ex}
            </button>
          ))}
        </div>

        {status === "error" && (
          <p role="alert" className="rounded-lg bg-red-50 p-4 text-sm text-red-800">
            {error}
          </p>
        )}

        {status === "done" && results.length === 0 && (
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Nenhum sermão encontrado para “{submitted}”. Tente outras palavras.
          </p>
        )}

        {status === "done" && results.length > 0 && (
          <p className="mb-3 text-xs text-slate-500 dark:text-slate-400">
            {results.length} {results.length === 1 ? "trecho" : "trechos"} · {tookMs} ms
          </p>
        )}

        <div className="space-y-3">
          {results.map((r) => (
            <SermonCard key={`${r.id}-${r.chunkIndex}`} result={r} query={submitted} />
          ))}
        </div>

        <SuggestionBox />
      </div>
    </div>
  );
}

function SuggestionBox() {
  const [text, setText] = useState("");
  const [sent, setSent] = useState(false);

  async function submit(e: FormEvent): Promise<void> {
    e.preventDefault();
    if (text.trim().length < 3) return;
    try {
      await sendSuggestion(text.trim());
      setSent(true);
      setText("");
    } catch {
      setSent(false);
    }
  }

  return (
    <section className="mt-12 border-t border-slate-200 pt-6 dark:border-slate-700">
      <h2 className="text-sm font-semibold">Sugestões</h2>
      {sent ? (
        <p className="mt-2 text-sm text-green-700 dark:text-green-400">
          Obrigado! Sua sugestão foi registrada.
        </p>
      ) : (
        <form onSubmit={submit} className="mt-2 flex gap-2">
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Faltou algum sermão? Conte para nós."
            aria-label="Sugestão"
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-800"
          />
          <button
            type="submit"
            className="rounded-lg border border-slate-300 px-4 py-2 text-sm dark:border-slate-600"
          >
            Enviar
          </button>
        </form>
      )}
    </section>
  );
}
