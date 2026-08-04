import { describe, expect, it } from "vitest";
import { SPOTIFY_LINKS_ALIVE_FROM, soundcloudUrl, spotifyUrl } from "../src/lib/audio-urls.ts";

/**
 * These URLs were verified to resolve (HTTP 200) against the live services and
 * cross-checked against the ipperegrinos track dump. They exist to catch a
 * regression of the bug where the SoundCloud channel segment was dropped and
 * every play link 404'd.
 */
const KNOWN = [
  {
    slug: "17-09-2021-a-lei-moral-e-a-vida-crista-piedade-e-nao-legalismo-1",
    date: "2021-09-17",
    episodeId: "1PR7EQBy9nxeCjlQlqxMS5",
    soundcloud:
      "https://soundcloud.com/ipperegrinos/17-09-2021-a-lei-moral-e-a-vida-crista-piedade-e-nao-legalismo-1",
    // Pre-2022: suppressed, see SPOTIFY_LINKS_ALIVE_FROM.
    spotify: null,
  },
  {
    slug: "19-03-2023-rute-3",
    date: "2023-03-19",
    episodeId: "7q9ozhxfNEXDkXkRhTT2nz",
    soundcloud: "https://soundcloud.com/ipperegrinos/19-03-2023-rute-3",
    spotify: "https://open.spotify.com/episode/7q9ozhxfNEXDkXkRhTT2nz",
  },
  {
    slug: "28-04-2024-efesios-61-3-filhos",
    date: "2024-04-28",
    episodeId: "3hYS4zZgngHyjhHg8F7aHs",
    soundcloud: "https://soundcloud.com/ipperegrinos/28-04-2024-efesios-61-3-filhos",
    spotify: "https://open.spotify.com/episode/3hYS4zZgngHyjhHg8F7aHs",
  },
];

describe("audio urls", () => {
  it.each(KNOWN)("rebuilds the canonical links for $slug", (k) => {
    expect(soundcloudUrl(k.slug)).toBe(k.soundcloud);
    expect(spotifyUrl(k.episodeId, k.date)).toBe(k.spotify);
  });

  it("keeps the channel segment in every SoundCloud link", () => {
    // The whole bug in one assertion: a track URL without a channel 404s.
    const url = soundcloudUrl("19-03-2023-rute-3");
    expect(url).toMatch(/^https:\/\/soundcloud\.com\/ipperegrinos\/[^/]+$/);
  });

  it("returns null for missing or blank suffixes", () => {
    for (const empty of [null, undefined, "", "   "]) {
      expect(soundcloudUrl(empty)).toBeNull();
      expect(spotifyUrl(empty, "2024-01-01")).toBeNull();
    }
  });

  describe("the pre-2022 Spotify suppression", () => {
    const id = "3hYS4zZgngHyjhHg8F7aHs";

    it("hides the link right up to the cutoff and shows it from the cutoff on", () => {
      expect(spotifyUrl(id, "2021-12-31")).toBeNull();
      expect(spotifyUrl(id, SPOTIFY_LINKS_ALIVE_FROM)).toBe(
        `https://open.spotify.com/episode/${id}`,
      );
    });

    it("hides the link when the date is missing or unparseable", () => {
      expect(spotifyUrl(id, null)).toBeNull();
      expect(spotifyUrl(id, "")).toBeNull();
    });

    it("never suppresses SoundCloud, which covers the whole corpus", () => {
      expect(soundcloudUrl("03-05-2020-mateus-517-20")).toBe(
        "https://soundcloud.com/ipperegrinos/03-05-2020-mateus-517-20",
      );
    });
  });

  it("does not double the channel if a suffix already carries it", () => {
    expect(soundcloudUrl("ipperegrinos/19-03-2023-rute-3")).toBe(
      "https://soundcloud.com/ipperegrinos/19-03-2023-rute-3",
    );
    expect(soundcloudUrl("/19-03-2023-rute-3")).toBe(
      "https://soundcloud.com/ipperegrinos/19-03-2023-rute-3",
    );
  });
});
