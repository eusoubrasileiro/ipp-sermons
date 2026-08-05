import { type RenderResult, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { App } from "./App.tsx";

/**
 * Shared fixtures for the behavioural tests.
 *
 * Every facet is addressable now, so anything rendering <App /> needs router
 * context; putting that here keeps each test file about behaviour rather than
 * setup, and gives the browse-page tests the same starting point.
 */
export const renderApp = (): RenderResult => render(<App />, { wrapper: MemoryRouter });

export const result = {
  id: "123",
  title: "17-09-2021 - A Lei Moral e a Vida Cristã",
  artist: "Pastor Alan Kleber",
  date: "2021-09-17",
  durationStr: "1:02:39",
  soundcloudUrl:
    "https://soundcloud.com/ipperegrinos/17-09-2021-a-lei-moral-e-a-vida-crista-piedade-e-nao-legalismo-1",
  spotifyUrl: "https://open.spotify.com/episode/1PR7EQBy9nxeCjlQlqxMS5",
  content: "a lei moral permanece como regra de vida para o cristão",
  score: 0.032,
  chunkIndex: 3,
};

export const okResponse = (body: unknown) =>
  ({ ok: true, json: async () => body }) as unknown as Response;

export const searchFor = async (text: string): Promise<void> => {
  await userEvent.type(screen.getByLabelText(/buscar nos sermões/i), text);
  await userEvent.click(screen.getByRole("button", { name: /^buscar$/i }));
};
