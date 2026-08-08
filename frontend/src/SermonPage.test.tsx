import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App.tsx";
import { okResponse, result } from "./test-fixtures.tsx";

/**
 * Reading a whole sermon.
 *
 * Driven through the real <App /> and the real route, because the thing most
 * likely to break is the wiring: a link that does not navigate, or a route the
 * SPA fallback swallows. The paragraph maths has its own unit tests in
 * lib/paragraphs.test.ts.
 */

const transcript = {
  id: "123",
  title: "17-09-2021 - A Lei Moral e a Vida Cristã",
  artist: "Pastor Alan Kleber",
  date: "2021-09-17",
  durationStr: "1:02:39",
  soundcloudUrl: "https://soundcloud.com/ipperegrinos/a-lei-moral",
  spotifyUrl: null,
  words: 6424,
  text: "Primeira frase da mensagem. A lei moral permanece como regra de vida. Última frase.",
};

const renderAt = (path: string) =>
  render(<App />, {
    wrapper: ({ children }) => <MemoryRouter initialEntries={[path]}>{children}</MemoryRouter>,
  });

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) =>
      url.includes("/transcript")
        ? okResponse(transcript)
        : okResponse({ livros: [], series: [], pregadores: [], datas: [], tipos: [], temas: [] }),
    ),
  );
});

describe("SermonPage", () => {
  it("renders the whole transcript", async () => {
    renderAt("/sermao/123");

    expect(
      await screen.findByRole("heading", { name: /a lei moral e a vida cristã/i }),
    ).toBeVisible();
    expect(screen.getByText(/primeira frase da mensagem/i)).toBeVisible();
    expect(screen.getByText(/última frase/i)).toBeVisible();
  });

  it("warns that the text is machine-transcribed", async () => {
    // A church archive misquoting a preacher is a real problem, not a UX nicety.
    renderAt("/sermao/123");
    expect(await screen.findByText(/transcrição automática/i)).toBeVisible();
  });

  it("offers the audio, which is still the primary artifact", async () => {
    renderAt("/sermao/123");
    const play = await screen.findByRole("link", { name: /soundcloud/i });
    expect(play).toHaveAttribute("href", transcript.soundcloudUrl);
  });

  it("states a reading time from the word count", async () => {
    // 6424 words at ~200 wpm.
    renderAt("/sermao/123");
    expect(await screen.findByText(/32 min de leitura/i)).toBeVisible();
  });

  it("highlights the search terms when arriving from a result", async () => {
    renderAt("/sermao/123?q=lei%20moral&trecho=0");

    const marks = await screen.findAllByText(/lei|moral/i, { selector: "mark" });
    expect(marks.length).toBeGreaterThan(0);
    expect(screen.getByText(/trecho que respondeu à sua busca/i)).toBeVisible();
  });

  it("opens plainly when there is no search behind it", async () => {
    renderAt("/sermao/123");

    await screen.findByText(/primeira frase da mensagem/i);
    expect(screen.queryByText(/trecho que respondeu à sua busca/i)).not.toBeInTheDocument();
    expect(document.querySelector("mark")).toBeNull();
  });

  it("sends the reader back to the search they came from", async () => {
    renderAt("/sermao/123?q=lei%20moral");
    expect(await screen.findByRole("link", { name: /voltar/i })).toHaveAttribute(
      "href",
      "/?q=lei%20moral",
    );
  });

  it("shows a recoverable error rather than a blank page", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          ({
            ok: false,
            json: async () => ({ error: "Sermão não encontrado." }),
          }) as unknown as Response,
      ),
    );
    renderAt("/sermao/999");

    expect(await screen.findByRole("alert")).toHaveTextContent(/sermão não encontrado/i);
    expect(screen.getByRole("button", { name: /tentar de novo/i })).toBeVisible();
  });

  it("is reachable from a search result, carrying the query and the matched chunk", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.includes("/transcript")) return okResponse(transcript);
        if (url.includes("/api/search")) {
          return okResponse({ query: "lei moral", results: [result], reranked: false, tookMs: 5 });
        }
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

    render(<App />, { wrapper: MemoryRouter });
    await userEvent.type(screen.getByLabelText(/buscar nos sermões/i), "lei moral");
    await userEvent.click(screen.getByRole("button", { name: /^buscar$/i }));

    const link = await screen.findByRole("link", { name: /ler a mensagem inteira/i });
    // chunkIndex 3 on the fixture; the reading page needs it to anchor.
    expect(link).toHaveAttribute("href", "/sermao/123?q=lei%20moral&trecho=3");

    await userEvent.click(link);
    await waitFor(() => expect(screen.getByText(/primeira frase da mensagem/i)).toBeVisible());
  });
});
