import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App.tsx";

/**
 * Behavioural tests against the real component, with fetch stubbed. They cover
 * what a visitor actually does: search, read a result, hit an error, find
 * nothing.
 */

const result = {
  id: "123",
  title: "17-09-2021 - A Lei Moral e a Vida Cristã",
  artist: "Pastor Alan Kleber",
  date: "2021-09-17",
  durationStr: "1:02:39",
  soundcloudUrl: "https://soundcloud.com/a-lei-moral",
  spotifyUrl: "https://open.spotify.com/episode/abc",
  content: "a lei moral permanece como regra de vida para o cristão",
  score: 0.032,
  chunkIndex: 3,
};

const okResponse = (body: unknown) => ({ ok: true, json: async () => body }) as unknown as Response;

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("shows results after a search", async () => {
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "lei moral", results: [result], reranked: false, tookMs: 120 }),
    );

    render(<App />);
    await userEvent.type(screen.getByLabelText(/o que você procura/i), "lei moral");
    await userEvent.click(screen.getByRole("button", { name: /buscar/i }));

    expect(await screen.findByText(/A Lei Moral e a Vida Cristã/)).toBeInTheDocument();
    expect(screen.getByText(/Pastor Alan Kleber/)).toBeInTheDocument();
  });

  it("renders both playback links", async () => {
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "lei", results: [result], reranked: false, tookMs: 10 }),
    );

    render(<App />);
    await userEvent.type(screen.getByLabelText(/o que você procura/i), "lei");
    await userEvent.click(screen.getByRole("button", { name: /buscar/i }));

    expect(await screen.findByRole("link", { name: /soundcloud/i })).toHaveAttribute(
      "href",
      result.soundcloudUrl,
    );
    expect(screen.getByRole("link", { name: /spotify/i })).toHaveAttribute(
      "href",
      result.spotifyUrl,
    );
  });

  it("formats the date for a Brazilian reader", async () => {
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "lei", results: [result], reranked: false, tookMs: 10 }),
    );

    render(<App />);
    await userEvent.type(screen.getByLabelText(/o que você procura/i), "lei");
    await userEvent.click(screen.getByRole("button", { name: /buscar/i }));

    expect(await screen.findByText(/17\/09\/2021/)).toBeInTheDocument();
  });

  it("tells the visitor when nothing matched", async () => {
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "xyz", results: [], reranked: false, tookMs: 8 }),
    );

    render(<App />);
    await userEvent.type(screen.getByLabelText(/o que você procura/i), "xyz");
    await userEvent.click(screen.getByRole("button", { name: /buscar/i }));

    expect(await screen.findByText(/Nenhum sermão encontrado/)).toBeInTheDocument();
  });

  it("surfaces a search failure without crashing", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      json: async () => ({ error: "A busca falhou. Tente novamente." }),
    } as unknown as Response);

    render(<App />);
    await userEvent.type(screen.getByLabelText(/o que você procura/i), "lei");
    await userEvent.click(screen.getByRole("button", { name: /buscar/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/falhou/i);
  });

  it("keeps the button disabled until the query is long enough", async () => {
    render(<App />);
    const button = screen.getByRole("button", { name: /buscar/i });
    expect(button).toBeDisabled();

    await userEvent.type(screen.getByLabelText(/o que você procura/i), "fé");
    expect(button).toBeEnabled();
  });

  it("runs a search when an example chip is clicked", async () => {
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "briga na igreja", results: [result], reranked: false, tookMs: 40 }),
    );

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "briga na igreja" }));

    await waitFor(() => expect(fetch).toHaveBeenCalled());
    const body = JSON.parse(String(vi.mocked(fetch).mock.calls[0]?.[1]?.body));
    expect(body.query).toBe("briga na igreja");
  });

  it("thanks the visitor after a suggestion is sent", async () => {
    vi.mocked(fetch).mockResolvedValue(okResponse({ ok: true }));

    render(<App />);
    await userEvent.type(screen.getByLabelText(/sugestão/i), "faltou o sermão de Tito 2");
    await userEvent.click(screen.getByRole("button", { name: /enviar/i }));

    expect(await screen.findByText(/Obrigado/)).toBeInTheDocument();
  });
});
