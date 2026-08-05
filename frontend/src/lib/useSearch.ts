import type { SearchResult } from "@ipp/shared";
import { useCallback, useEffect, useRef, useState } from "react";
import { searchSermons } from "../api.ts";
import { parseFilters } from "./facet-params.ts";

/**
 * Runs the search the URL describes.
 *
 * Keyed on the query and the serialised filters rather than on a filters
 * object, so React can compare them: a fresh object literal every render would
 * re-fire the effect forever. Both come straight from the address bar, which
 * makes the URL the only place search state lives.
 *
 * Retrying re-runs the same call rather than bumping a counter in the
 * dependency list, so there is exactly one description of how a search is run.
 */
type SearchStatus = "idle" | "loading" | "done" | "error";

type SearchState = {
  status: SearchStatus;
  results: SearchResult[];
  tookMs: number;
  error: string;
};

const IDLE: SearchState = { status: "idle", results: [], tookMs: 0, error: "" };

export function useSearch(query: string, filterQuery: string): SearchState & { retry: () => void } {
  const [state, setState] = useState<SearchState>(IDLE);
  // Only the newest request may write: removing a chip while a slower search
  // is still in flight would otherwise land the filtered results afterwards.
  const latest = useRef(0);

  const run = useCallback(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setState(IDLE);
      return;
    }

    const token = latest.current + 1;
    latest.current = token;
    setState((prev) => ({ ...prev, status: "loading", error: "" }));

    searchSermons(trimmed, parseFilters(new URLSearchParams(filterQuery)))
      .then((res) => {
        if (token === latest.current) {
          setState({ status: "done", results: res.results, tookMs: res.tookMs, error: "" });
        }
      })
      .catch((err: unknown) => {
        if (token === latest.current) {
          setState({
            status: "error",
            results: [],
            tookMs: 0,
            error: err instanceof Error ? err.message : "A busca falhou.",
          });
        }
      });
  }, [query, filterQuery]);

  useEffect(run, [run]);

  return { ...state, retry: run };
}
