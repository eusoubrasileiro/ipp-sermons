import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { okResponse, renderApp, result, searchFor } from "./test-fixtures.tsx";

/**
 * Behavioural tests against the real component, with fetch stubbed. They cover
 * what a visitor actually does: search, read a result, play it, hit an error,
 * find nothing.
 */

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

    renderApp();
    await searchFor("lei moral");

    // The date prefix the corpus carries is dropped from the heading.
    expect(
      await screen.findByRole("heading", { name: "A Lei Moral e a Vida Cristã" }),
    ).toBeVisible();
    expect(screen.getByText(/Pastor Alan Kleber/)).toBeInTheDocument();
  });

  it("shows SoundCloud alone when the API withholds the Spotify link", async () => {
    // Pre-2022 episodes were retired upstream, so the API returns a null
    // spotifyUrl for them -- the card must still offer a way to listen.
    const old = { ...result, date: "2020-05-03", spotifyUrl: null };
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "lei", results: [old], reranked: false, tookMs: 10 }),
    );

    renderApp();
    await searchFor("lei");

    expect(await screen.findByRole("link", { name: /no SoundCloud/i })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /no Spotify/i })).not.toBeInTheDocument();
  });

  it("formats the date for a Brazilian reader", async () => {
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "lei", results: [result], reranked: false, tookMs: 10 }),
    );

    renderApp();
    await searchFor("lei");

    expect(await screen.findByText("17 set 2021")).toBeInTheDocument();
  });

  it("highlights the query inside the excerpt, accents and all", async () => {
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "cristao", results: [result], reranked: false, tookMs: 10 }),
    );

    renderApp();
    await searchFor("cristao");

    const mark = await screen.findByText("cristão");
    expect(mark.tagName).toBe("MARK");
  });

  it("shows one card per sermon when several passages match", async () => {
    const second = { ...result, chunkIndex: 9, content: "outra passagem sobre a lei" };
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "lei", results: [result, second], reranked: true, tookMs: 30 }),
    );

    renderApp();
    await searchFor("lei");

    expect(
      await screen.findAllByRole("heading", { name: "A Lei Moral e a Vida Cristã" }),
    ).toHaveLength(1);
    await userEvent.click(screen.getByRole("button", { name: /mais 1 passagem/i }));
    // The excerpt is split around the highlighted term, so match the plain part.
    expect(screen.getByText(/outra passagem sobre/)).toBeInTheDocument();
  });

  it("tells the visitor when nothing matched", async () => {
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "xyz", results: [], reranked: false, tookMs: 8 }),
    );

    renderApp();
    await searchFor("xyz");

    expect(await screen.findByText(/Nenhum sermão encontrado/)).toBeInTheDocument();
  });

  it("surfaces a search failure and offers a retry", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      json: async () => ({ error: "A busca falhou. Tente novamente." }),
    } as unknown as Response);

    renderApp();
    await searchFor("lei");

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/falhou/i);
    expect(screen.getByRole("button", { name: /tentar de novo/i })).toBeInTheDocument();
  });

  it("keeps the button disabled until the query is long enough", async () => {
    renderApp();
    const button = screen.getByRole("button", { name: /^buscar$/i });
    expect(button).toBeDisabled();

    await userEvent.type(screen.getByLabelText(/buscar nos sermões/i), "fé");
    expect(button).toBeEnabled();
  });

  it("runs a search when an example chip is clicked", async () => {
    vi.mocked(fetch).mockResolvedValue(
      okResponse({ query: "briga na igreja", results: [result], reranked: false, tookMs: 40 }),
    );

    renderApp();
    await userEvent.click(screen.getByRole("button", { name: "briga na igreja" }));

    await waitFor(() => expect(fetch).toHaveBeenCalled());
    const body = JSON.parse(String(vi.mocked(fetch).mock.calls[0]?.[1]?.body));
    expect(body.query).toBe("briga na igreja");
  });

  it("thanks the visitor after a suggestion is sent", async () => {
    vi.mocked(fetch).mockResolvedValue(okResponse({ ok: true }));

    renderApp();
    await userEvent.type(screen.getByLabelText(/sugestão/i), "faltou o sermão de Tito 2");
    await userEvent.click(screen.getByRole("button", { name: /enviar/i }));

    expect(await screen.findByText(/Obrigado/)).toBeInTheDocument();
  });
});
