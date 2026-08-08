import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App.tsx";
import { okResponse } from "./test-fixtures.tsx";

/**
 * Whose job the browser tab is, now that the server writes a title into it.
 *
 * Before the prerender the title was one constant for the whole site, so
 * nothing had to maintain it. Now a page arrives naming the sermon it shows —
 * which is right until the visitor clicks a link and React changes the page
 * underneath a title nobody updated. These pin the two halves: the server's
 * title survives on the page it was written for, and it does not follow the
 * visitor anywhere else.
 */

const SITE = "Sermões IPP — Igreja Presbiteriana Peregrinos";

const transcript = {
  id: "123",
  title: "17-09-2021 - A Lei Moral e a Vida Cristã",
  artist: "Pastor Alan Kleber",
  date: "2021-09-17",
  durationStr: "1:02:39",
  soundcloudUrl: "https://soundcloud.com/ipperegrinos/a-lei-moral",
  spotifyUrl: null,
  words: 6424,
  text: "Primeira frase da mensagem. Última frase.",
};

const renderAt = (path: string) =>
  render(<App />, {
    wrapper: ({ children }) => <MemoryRouter initialEntries={[path]}>{children}</MemoryRouter>,
  });

beforeEach(() => {
  document.title = SITE;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url.includes("/transcript")) return okResponse(transcript);
      if (url.includes("/api/sermons")) return okResponse({ sermons: [], total: 0, pagina: 1 });
      return okResponse({
        livros: [],
        series: [],
        pregadores: [],
        datas: [],
        tipos: [],
        temas: [],
      });
    }),
  );
});

describe("the browser tab", () => {
  it("names the sermon being read", async () => {
    renderAt("/sermao/123");

    // The same string the server wrote into <title>, so the tab does not
    // change under the reader when React takes over.
    await waitFor(() =>
      expect(document.title).toBe("A Lei Moral e a Vida Cristã — Pastor Alan Kleber"),
    );
  });

  it("stops naming it once the visitor navigates away", async () => {
    // Without this the tab keeps advertising a sermon the visitor left, which
    // is worse than the generic title it used to always show.
    document.title = "A Lei Moral e a Vida Cristã — Pastor Alan Kleber";
    renderAt("/sermao/123");

    await screen.findByRole("heading", { name: /a lei moral/i });
    await userEvent.click(screen.getByRole("link", { name: "Temas" }));

    await waitFor(() => expect(document.title).toBe(SITE));
  });

  it("leaves the title the server wrote for the page actually landed on", async () => {
    // A crawler-visible title on /biblia/tito must survive hydration; only a
    // later navigation may replace it.
    document.title = "Tito — Sermões IPP";
    renderAt("/biblia/tito");

    await waitFor(() => expect(screen.queryByText(/carregando/i)).not.toBeInTheDocument());
    expect(document.title).toBe("Tito — Sermões IPP");
  });
});
