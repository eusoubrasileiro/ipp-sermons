import { describe, expect, it } from "vitest";
import { soundcloudUrl, spotifyUrl } from "./audio-urls.ts";

/**
 * These URLs were verified to resolve (HTTP 200) against the live services and
 * cross-checked against the ipperegrinos track dump. They exist to catch a
 * regression of the bug where the SoundCloud channel segment was dropped and
 * every play link 404'd.
 *
 * `alive` is what `data/facets/spotify_episodes.csv` records for the episode.
 * The 2021 row is the case the old date cutoff got wrong: it is still inside
 * the podcast feed's 500-item window and its episode answers 200, but a
 * `date < 2022-01-01` rule hid it anyway.
 */
const KNOWN = [
  {
    slug: "17-09-2021-a-lei-moral-e-a-vida-crista-piedade-e-nao-legalismo-1",
    alive: true,
    episodeId: "1PR7EQBy9nxeCjlQlqxMS5",
    soundcloud:
      "https://soundcloud.com/ipperegrinos/17-09-2021-a-lei-moral-e-a-vida-crista-piedade-e-nao-legalismo-1",
    spotify: "https://open.spotify.com/episode/1PR7EQBy9nxeCjlQlqxMS5",
  },
  {
    slug: "19-03-2023-rute-3",
    alive: true,
    episodeId: "7q9ozhxfNEXDkXkRhTT2nz",
    soundcloud: "https://soundcloud.com/ipperegrinos/19-03-2023-rute-3",
    spotify: "https://open.spotify.com/episode/7q9ozhxfNEXDkXkRhTT2nz",
  },
  {
    slug: "03-05-2020-mateus-517-20",
    alive: false,
    episodeId: "5hDavifsOIrEuhIaJPL0KE",
    soundcloud: "https://soundcloud.com/ipperegrinos/03-05-2020-mateus-517-20",
    // Aged out of the 500-item feed window; verified 404 on Spotify.
    spotify: null,
  },
];

describe("audio urls", () => {
  it.each(KNOWN)("rebuilds the canonical links for $slug", (k) => {
    expect(soundcloudUrl(k.slug)).toBe(k.soundcloud);
    expect(spotifyUrl(k.episodeId, k.alive)).toBe(k.spotify);
  });

  it("keeps the channel segment in every SoundCloud link", () => {
    // The whole bug in one assertion: a track URL without a channel 404s.
    const url = soundcloudUrl("19-03-2023-rute-3");
    expect(url).toMatch(/^https:\/\/soundcloud\.com\/ipperegrinos\/[^/]+$/);
  });

  it("returns null for missing or blank suffixes", () => {
    for (const empty of [null, undefined, "", "   "]) {
      expect(soundcloudUrl(empty)).toBeNull();
      expect(spotifyUrl(empty, true)).toBeNull();
    }
  });

  describe("suppressing episodes that aged out of the podcast feed", () => {
    const id = "3hYS4zZgngHyjhHg8F7aHs";

    it("shows the link only while the episode is still in the feed", () => {
      expect(spotifyUrl(id, true)).toBe(`https://open.spotify.com/episode/${id}`);
      expect(spotifyUrl(id, false)).toBeNull();
    });

    it("suppresses the link when liveness is unknown rather than guessing", () => {
      // A sermon indexed before check:spotify ran has no answer yet; a dead
      // link is worse than none, so the absent case must not fall through.
      expect(spotifyUrl(id, null)).toBeNull();
      expect(spotifyUrl(id, undefined)).toBeNull();
    });

    it("does not use the sermon date, which was only ever a proxy", () => {
      // The old rule hid every pre-2022 episode. This 2021 one still resolves.
      expect(spotifyUrl("1PR7EQBy9nxeCjlQlqxMS5", true)).toBe(
        "https://open.spotify.com/episode/1PR7EQBy9nxeCjlQlqxMS5",
      );
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
