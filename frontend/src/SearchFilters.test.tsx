import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { routeTo, searchBodies, stubNoResults } from "./browse-fixtures.tsx";

/**
 * Filters composing with the search — the one thing this site does that
 * Desiring God does not.
 *
 * Driven at real URLs, because the whole design puts the filter state in the
 * address bar: a filtered search has to survive a reload and a forwarded link.
 */

const openPicker = async (dimension: string): Promise<void> => {
  await userEvent.click(screen.getByRole("button", { name: /adicionar filtro/i }));
  await userEvent.click(await screen.findByRole("button", { name: dimension }));
};

describe("filtros compostos", () => {
  it("runs a filtered search straight from the URL", async () => {
    routeTo("/?q=casamento&livros=efesios&capitulo=5");

    await waitFor(() => expect(searchBodies()).toHaveLength(1));
    expect(searchBodies()[0]).toMatchObject({
      query: "casamento",
      filtros: { livros: ["efesios"], capitulo: 5 },
    });
  });

  it("names the filter in Portuguese, not by slug", async () => {
    routeTo("/?q=casamento&livros=efesios&capitulo=5");

    // The chapter rides on the book rather than becoming a chip of its own.
    expect(await screen.findByText("Efésios 5")).toBeVisible();
  });

  it("re-runs the search without a filter once its chip is removed", async () => {
    routeTo("/?q=casamento&livros=efesios");
    await waitFor(() => expect(searchBodies()).toHaveLength(1));

    await userEvent.click(screen.getByRole("button", { name: /remover filtro efésios/i }));

    await waitFor(() => expect(searchBodies()).toHaveLength(2));
    expect(searchBodies()[1]?.filtros).toEqual({});
    expect(screen.queryByText("Efésios")).not.toBeInTheDocument();
  });

  it("offers only the options that still leave results", async () => {
    routeTo("/?q=casamento");
    await openPicker("Pregador");

    // The stubbed counts have Bruno Melo at 2 and no Lucas Antunes at all.
    expect(await screen.findByRole("button", { name: /bruno melo, 2 sermões/i })).toBeVisible();
    expect(screen.queryByRole("button", { name: /lucas antunes/i })).not.toBeInTheDocument();
  });

  it("adds the chosen filter to the search", async () => {
    routeTo("/?q=casamento");
    await waitFor(() => expect(searchBodies()).toHaveLength(1));

    await openPicker("Pregador");
    await userEvent.click(await screen.findByRole("button", { name: /bruno melo, 2 sermões/i }));

    await waitFor(() => expect(searchBodies()).toHaveLength(2));
    expect(searchBodies()[1]?.filtros).toEqual({ pregadores: ["Reverendo Bruno Melo"] });
  });

  it("filters by year as a date range, because that is what the API takes", async () => {
    routeTo("/?q=casamento");
    await waitFor(() => expect(searchBodies()).toHaveLength(1));

    await openPicker("Ano");
    await userEvent.click(await screen.findByRole("button", { name: /2024, 2 sermões/i }));

    await waitFor(() => expect(searchBodies()).toHaveLength(2));
    expect(searchBodies()[1]?.filtros).toEqual({ de: "2024-01-01", ate: "2024-12-31" });
  });

  it("says so when a dimension has nothing left to offer", async () => {
    routeTo("/?q=casamento");
    await openPicker("Tema");

    // No sermon is labelled yet, so the column is empty rather than absent --
    // an option list that silently vanishes reads as a broken page.
    expect(await screen.findByText(/nada a acrescentar/i)).toBeVisible();
  });

  it("lists the archive instead of searching when there is a filter but no query", async () => {
    // Someone can reach the search page and pick a filter before typing
    // anything; the page must show what it selected rather than sit idle.
    routeTo("/?tipos=ebd");

    await waitFor(() =>
      expect(
        vi
          .mocked(fetch)
          .mock.calls.some(([url]) => String(url).startsWith("/api/sermons?tipos=ebd")),
      ).toBe(true),
    );
    expect(searchBodies()).toHaveLength(0);
  });

  it("offers a way out when the filters leave nothing", async () => {
    stubNoResults();
    routeTo("/?q=casamento&livros=efesios");

    await userEvent.click(await screen.findByRole("button", { name: /limpar os filtros/i }));

    await waitFor(() => expect(searchBodies()).toHaveLength(2));
    expect(searchBodies()[1]?.filtros).toEqual({});
  });
});
