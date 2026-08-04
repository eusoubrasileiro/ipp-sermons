import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PlayLinks } from "./PlayLinks.tsx";

/**
 * The play links are the product: a result nobody can open is worthless. These
 * pin the link targets, the accessible names, and every combination of the two
 * platforms being available.
 */

const soundcloud = "https://soundcloud.com/ipperegrinos/19-03-2023-rute-3";
const spotify = "https://open.spotify.com/episode/7q9ozhxfNEXDkXkRhTT2nz";

describe("PlayLinks", () => {
  it("renders both platforms with the sermon in the accessible name", () => {
    render(<PlayLinks title="Rute 3" soundcloudUrl={soundcloud} spotifyUrl={spotify} />);

    expect(screen.getByRole("link", { name: 'Ouvir "Rute 3" no SoundCloud' })).toHaveAttribute(
      "href",
      soundcloud,
    );
    expect(screen.getByRole("link", { name: 'Ouvir "Rute 3" no Spotify' })).toHaveAttribute(
      "href",
      spotify,
    );
  });

  it("renders SoundCloud alone when the Spotify link is withheld", () => {
    // Pre-2022 episodes were retired upstream; SoundCloud still covers them.
    render(<PlayLinks title="Rute 3" soundcloudUrl={soundcloud} spotifyUrl={null} />);

    expect(screen.getByRole("link", { name: /no SoundCloud/ })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /no Spotify/ })).not.toBeInTheDocument();
  });

  it("opens in a new tab without leaking the referrer", () => {
    render(<PlayLinks title="Rute 3" soundcloudUrl={soundcloud} spotifyUrl={spotify} />);

    for (const link of screen.getAllByRole("link")) {
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noreferrer");
    }
  });

  it("says so instead of rendering an empty row when there is no audio at all", () => {
    render(<PlayLinks title="Rute 3" soundcloudUrl={null} spotifyUrl={null} />);

    expect(screen.queryAllByRole("link")).toHaveLength(0);
    expect(screen.getByText(/Áudio indisponível/)).toBeInTheDocument();
  });
});
