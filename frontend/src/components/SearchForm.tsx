import { type FormEvent, useRef } from "react";
import { SearchIcon } from "./Icons.tsx";

/**
 * The search box, sticky at the top.
 *
 * Most of the congregation arrives on a phone: once results scroll past, the
 * only way back to the query would be scrolling up, so the box stays put.
 */

const EXAMPLES = [
  "briga na igreja",
  "justificação pela fé",
  "batismo infantil",
  "Eclesiastes",
  "ansiedade",
];

export function SearchForm({
  query,
  onQueryChange,
  onSearch,
  loading,
  showExamples,
}: {
  query: string;
  onQueryChange: (q: string) => void;
  onSearch: (q: string) => void;
  loading: boolean;
  /** Hidden once results are on screen: the sticky bar then eats a fifth of a phone. */
  showExamples: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  function submit(e: FormEvent): void {
    e.preventDefault();
    onSearch(query);
    // Dismiss the phone keyboard so the first result is visible immediately.
    inputRef.current?.blur();
  }

  return (
    <div className="sticky top-0 z-10 -mx-4 border-b border-border/60 bg-background px-4 pb-3 pt-4 backdrop-blur supports-[backdrop-filter]:bg-background/90">
      <form onSubmit={submit}>
        <label htmlFor="q" className="sr-only">
          Buscar nos sermões
        </label>
        <div className="flex gap-2">
          <div className="relative flex-1">
            <SearchIcon className="pointer-events-none absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
            <input
              id="q"
              ref={inputRef}
              type="search"
              value={query}
              onChange={(e) => onQueryChange(e.target.value)}
              placeholder="Ex.: batismo infantil"
              enterKeyHint="search"
              autoComplete="off"
              className="h-12 w-full rounded-lg border border-input bg-card pl-10 pr-3 text-base text-card-foreground placeholder:text-muted-foreground focus:border-ring"
            />
          </div>
          <button
            type="submit"
            disabled={loading || query.trim().length < 2}
            className="h-12 shrink-0 rounded-lg bg-primary px-5 font-semibold text-primary-foreground shadow-sm transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Buscando…" : "Buscar"}
          </button>
        </div>
      </form>

      <div className={`mt-2 flex-wrap items-center gap-1.5 ${showExamples ? "flex" : "hidden"}`}>
        <span className="text-xs text-muted-foreground">Exemplos:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            disabled={loading}
            onClick={() => {
              onQueryChange(ex);
              onSearch(ex);
            }}
            className="rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground transition hover:bg-accent hover:text-accent-foreground disabled:opacity-50"
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
