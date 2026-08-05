import { type RenderResult, render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, vi } from "vitest";
import { App } from "./App.tsx";
import { COUNTS, FACETS, SEARCH_RESULT, SERMON } from "./facet-fixtures.ts";
import { clearFacetsCache } from "./lib/useFacets.ts";
import { okResponse } from "./test-fixtures.tsx";

/**
 * The browse side: the five index pages, the listings they link to, and the
 * filters that compose with the search.
 *
 * Driven through the real router at a real URL, because addressability is the
 * feature -- a member sends /biblia/efesios/5 to someone and it has to open
 * there.
 */

export const routeTo = (path: string): RenderResult =>
  render(<App />, {
    wrapper: ({ children }) => <MemoryRouter initialEntries={[path]}>{children}</MemoryRouter>,
  });

/** Every /api/search body the stub saw, newest last. */
export const searchBodies = (): { query: string; filtros?: Record<string, unknown> }[] =>
  vi
    .mocked(fetch)
    .mock.calls.filter(([url]) => String(url) === "/api/search")
    .map(([, init]) => JSON.parse(String(init?.body)));

function stubBrowseFetch(results: unknown[] = [SEARCH_RESULT]): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      // Order matters: /api/facets/counts is a prefix match of /api/facets.
      if (url.startsWith("/api/facets/counts")) return okResponse(COUNTS);
      if (url.startsWith("/api/facets")) return okResponse(FACETS);
      if (url.startsWith("/api/sermons")) {
        return okResponse({ total: 13, sermons: [SERMON], pagina: 1 });
      }
      if (url.startsWith("/api/search")) {
        return okResponse({ query: "q", results, reranked: false, tookMs: 12 });
      }
      return okResponse({});
    }),
  );
}

/** Re-stubs with an empty result set, for the "nothing matched" paths. */
export const stubNoResults = (): void => stubBrowseFetch([]);

beforeEach(() => {
  clearFacetsCache();
  stubBrowseFetch();
});

afterEach(() => {
  vi.unstubAllGlobals();
});
