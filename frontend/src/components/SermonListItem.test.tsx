import { render as rtlRender, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { toBrowseQuery } from "../api.ts";
import { SermonListItem } from "./SermonListItem.tsx";

/** The title links to the reading page now, so every render needs a router. */
const render = (ui: ReactElement) => rtlRender(ui, { wrapper: MemoryRouter });

const base = {
  id: "1",
  title: "17-03-2024 - Efésios 5.22-33 - O casamento",
  displayTitle: "Efésios 5.22-33 - O casamento",
  artist: "Reverendo Bruno Melo",
  date: "2024-03-17T00:00:00.000Z",
  durationStr: "0:48:25",
  scSuffixUrl: "casamento",
  spSuffixUrl: "abc",
  spotifyAlive: true,
  serviceType: "culto",
  seriesPart: null as number | null,
  series: null as { slug: string; name: string } | null,
  scriptures: [{ bookSlug: "efesios", chapter: 5, book: { name: "Efésios" } }],
};

describe("SermonListItem", () => {
  it("shows the service type, date, preacher and runtime", () => {
    render(<SermonListItem sermon={base} />);
    expect(screen.getByText(/Culto · 17 mar 2024 · Reverendo Bruno Melo · 48:25/)).toBeVisible();
  });

  it("numbers a lesson inside its course", () => {
    render(
      <SermonListItem
        sermon={{ ...base, seriesPart: 3, series: { slug: "reis", name: "O Livro dos Reis" } }}
      />,
    );
    expect(screen.getByRole("heading", { name: /3\./ })).toBeVisible();
    expect(screen.getByText(/O Livro dos Reis/)).toBeVisible();
  });

  it("falls back to the raw title when there is no display title", () => {
    render(<SermonListItem sermon={{ ...base, displayTitle: null }} />);
    // The leading date belongs in the meta line, not the heading.
    expect(screen.getByRole("heading", { name: /^Efésios 5.22-33/ })).toBeVisible();
  });

  it("suppresses the Spotify link once the episode has left the podcast feed", () => {
    // The browse path used to hand-copy the backend's date cutoff; it now
    // reads the same derived flag the search path does.
    render(<SermonListItem sermon={{ ...base, spotifyAlive: false }} />);
    expect(screen.getByRole("link", { name: /SoundCloud/ })).toBeVisible();
    expect(screen.queryByRole("link", { name: /Spotify/ })).not.toBeInTheDocument();
  });

  it("keeps the Spotify link for an old sermon whose episode is still live", () => {
    // The old date rule hid this; feed membership, not age, decides.
    render(
      <SermonListItem sermon={{ ...base, date: "2021-05-02T00:00:00.000Z", spotifyAlive: true }} />,
    );
    expect(screen.getByRole("link", { name: /Spotify/ })).toBeVisible();
  });

  it("says so when a sermon has no audio at all", () => {
    render(<SermonListItem sermon={{ ...base, scSuffixUrl: null, spSuffixUrl: null }} />);
    expect(screen.getByText(/Áudio indisponível/)).toBeVisible();
  });

  it("renders a sermon with no passage and no series", () => {
    render(<SermonListItem sermon={{ ...base, scriptures: [], serviceType: null }} />);
    expect(screen.getByRole("heading", { name: /O casamento/ })).toBeVisible();
  });
});

describe("toBrowseQuery", () => {
  it("keeps only the facets that were set", () => {
    expect(toBrowseQuery({ livros: "efesios", capitulo: 5, temas: undefined, series: "" })).toBe(
      "livros=efesios&capitulo=5",
    );
  });

  it("is empty when nothing is selected", () => {
    expect(toBrowseQuery({})).toBe("");
  });
});
