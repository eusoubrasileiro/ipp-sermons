import { describe, expect, it } from "vitest";
import {
  assertDeadFractionSane,
  assertFeedSane,
  fetchLiveness,
  type LivenessRow,
  livenessRows,
  MIN_FEED_ITEMS,
  parseFeedSlugs,
  parseLivenessCsv,
} from "../src/lib/podcast-feed.ts";

/**
 * A trimmed copy of the real feed's shape, including the parts that must NOT be
 * read as track links: the channel `<link>`, the `feeds.soundcloud.com` stream
 * enclosure, and the `i1.sndcdn.com` artwork.
 */
const FEED = `<?xml version="1.0" encoding="utf-8"?>
<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" version="2.0">
  <channel>
    <title>Igreja Presbiteriana Peregrinos</title>
    <link>http://www.ipperegrinos.com</link>
    <item>
      <guid isPermaLink="false">tag:soundcloud,2010:tracks/2373669446</guid>
      <title>Eclesiastes 5:1 - Rev. Lucas Antunes</title>
      <link>https://soundcloud.com/ipperegrinos/eclesiastes-5-1-eclesiastes-5</link>
      <enclosure type="audio/mpeg" url="https://feeds.soundcloud.com/stream/2373669446-ipperegrinos-eclesiastes.mp3" length="79164856" />
      <itunes:image href="https://i1.sndcdn.com/artworks-naUMkH0d5CfhjD3k-t3000x3000.png" />
    </item>
    <item>
      <title>02-05-2021 - Gênesis 11.10-30</title>
      <link>https://soundcloud.com/ipperegrinos/02-05-2021-genesis-1110-30</link>
      <enclosure type="audio/mpeg" url="https://feeds.soundcloud.com/stream/1043493988-ipperegrinos-genesis.mp3" length="44346130" />
    </item>
  </channel>
</rss>`;

const sermon = (id: string, sc: string | null, sp: string | null) => ({
  id,
  scSuffixUrl: sc,
  spSuffixUrl: sp,
});

describe("parseFeedSlugs", () => {
  it("extracts the track slug from every item link", () => {
    expect(parseFeedSlugs(FEED)).toEqual([
      "eclesiastes-5-1-eclesiastes-5",
      "02-05-2021-genesis-1110-30",
    ]);
  });

  it("ignores the channel link, the stream enclosure and the artwork host", () => {
    const slugs = parseFeedSlugs(FEED);
    expect(slugs.some((s) => s.includes("sndcdn") || s.includes("stream"))).toBe(false);
    expect(slugs).not.toContain("ipperegrinos.com");
  });

  it("de-duplicates a track that appears twice", () => {
    const doubled = FEED.replace(
      "</channel>",
      "<item><link>https://soundcloud.com/ipperegrinos/02-05-2021-genesis-1110-30</link></item></channel>",
    );
    expect(parseFeedSlugs(doubled)).toHaveLength(2);
  });

  it("returns nothing for an empty or non-XML body", () => {
    expect(parseFeedSlugs("")).toEqual([]);
    expect(parseFeedSlugs("<html><body>502 Bad Gateway</body></html>")).toEqual([]);
  });
});

describe("assertFeedSane", () => {
  it("rejects an empty feed rather than marking the whole corpus dead", () => {
    // The 3am failure this exists to stop: a flake returns 0 items, every
    // episode is written as dead, and the smoke test only counts sermons.
    expect(() => assertFeedSane([])).toThrow(/0 items/);
  });

  it("rejects a feed truncated below the floor", () => {
    expect(() => assertFeedSane(Array.from({ length: 5 }, (_, i) => `s${i}`))).toThrow(/too few/i);
  });

  it("accepts a feed at the floor", () => {
    const full = Array.from({ length: MIN_FEED_ITEMS }, (_, i) => `slug-${i}`);
    expect(() => assertFeedSane(full)).not.toThrow();
  });
});

describe("livenessRows", () => {
  const live = new Set(["in-feed-a", "in-feed-b"]);

  it("marks a sermon alive when its SoundCloud slug is still in the feed", () => {
    const rows = livenessRows([sermon("1", "in-feed-a", "spid1")], live, "2026-08-07");
    expect(rows).toEqual([
      { sermon_id: "1", spotify_id: "spid1", alive: true, checked_at: "2026-08-07" },
    ]);
  });

  it("marks a sermon dead once its episode has aged out of the 500-item window", () => {
    const rows = livenessRows([sermon("2", "aged-out", "spid2")], live, "2026-08-07");
    expect(rows[0]?.alive).toBe(false);
  });

  it("skips sermons that never had a Spotify episode at all", () => {
    // "no episode" and "episode retired" are different facts; only the second
    // belongs in this file.
    expect(livenessRows([sermon("3", "in-feed-a", null)], live, "2026-08-07")).toEqual([]);
    expect(livenessRows([sermon("4", "in-feed-a", "  ")], live, "2026-08-07")).toEqual([]);
  });

  it("tolerates a suffix carrying a leading slash or the channel prefix", () => {
    const rows = livenessRows(
      [sermon("5", "ipperegrinos/in-feed-b", "spid5"), sermon("6", "/in-feed-a", "spid6")],
      live,
      "2026-08-07",
    );
    expect(rows.map((r) => r.alive)).toEqual([true, true]);
  });

  it("marks a sermon with no SoundCloud slug dead rather than guessing", () => {
    expect(livenessRows([sermon("7", null, "spid7")], live, "2026-08-07")[0]?.alive).toBe(false);
  });
});

describe("assertDeadFractionSane", () => {
  const rows = (deadCount: number, total: number): LivenessRow[] =>
    Array.from({ length: total }, (_, i) => ({
      sermon_id: `s${i}`,
      spotify_id: `sp${i}`,
      alive: i >= deadCount,
      checked_at: "2026-08-07",
    }));

  it("accepts the slow drift of a rolling window", () => {
    expect(() => assertDeadFractionSane(rows(120, 500), rows(130, 500))).not.toThrow();
  });

  it("rejects a collapse that would blank most of the archive", () => {
    expect(() => assertDeadFractionSane(rows(120, 500), rows(480, 500))).toThrow(/dead fraction/i);
  });

  it("accepts any result when there is no committed file to compare against", () => {
    expect(() => assertDeadFractionSane([], rows(120, 500))).not.toThrow();
  });

  it("never objects to episodes coming back to life", () => {
    expect(() => assertDeadFractionSane(rows(400, 500), rows(10, 500))).not.toThrow();
  });
});

describe("parseLivenessCsv", () => {
  it("reads a committed file back into rows", () => {
    expect(
      parseLivenessCsv([
        { sermon_id: "1", spotify_id: "sp1", alive: "true", checked_at: "2026-08-07" },
        { sermon_id: "2", spotify_id: "sp2", alive: "false", checked_at: "2026-08-07" },
      ]),
    ).toEqual([
      { sermon_id: "1", spotify_id: "sp1", alive: true, checked_at: "2026-08-07" },
      { sermon_id: "2", spotify_id: "sp2", alive: false, checked_at: "2026-08-07" },
    ]);
  });

  it("reads anything but the literal true as dead", () => {
    expect(parseLivenessCsv([{ sermon_id: "1", spotify_id: "s", alive: "" }])[0]?.alive).toBe(
      false,
    );
  });
});

describe("fetchLiveness", () => {
  const feedOf = (slugs: string[]) =>
    `<rss><channel>${slugs
      .map((s) => `<item><link>https://soundcloud.com/ipperegrinos/${s}</link></item>`)
      .join("")}</channel></rss>`;

  const ok = (body: string) =>
    (async () => new Response(body, { status: 200 })) as unknown as typeof fetch;

  const many = Array.from({ length: 200 }, (_, i) => `slug-${i}`);
  const sermons = [
    sermon("1", "slug-0", "spid1"),
    sermon("2", "slug-1", "spid2"),
    sermon("3", "gone", "spid3"),
  ];

  it("returns a row per episode and the feed size", async () => {
    const out = await fetchLiveness({
      fetchFn: ok(feedOf(many)),
      sermons,
      previous: [],
      checkedAt: "2026-08-07",
    });
    expect(out.feedItems).toBe(200);
    expect(out.rows.map((r) => r.alive)).toEqual([true, true, false]);
  });

  it("refuses to write when the feed request fails", async () => {
    const failing = (async () => new Response("nope", { status: 502 })) as unknown as typeof fetch;
    await expect(
      fetchLiveness({ fetchFn: failing, sermons, previous: [], checkedAt: "2026-08-07" }),
    ).rejects.toThrow(/HTTP 502/);
  });

  it("refuses a feed that came back short rather than marking the archive dead", async () => {
    await expect(
      fetchLiveness({
        fetchFn: ok(feedOf(["slug-0"])),
        sermons,
        previous: [],
        checkedAt: "2026-08-07",
      }),
    ).rejects.toThrow(/too few/i);
  });

  it("refuses a run that would blank most of the archive", async () => {
    // Every sermon drops out of the feed at once: the feed itself looks healthy,
    // so only the dead-fraction guard catches it.
    const previous = sermons.map((s) => ({
      sermon_id: s.id,
      spotify_id: s.spSuffixUrl ?? "",
      alive: true,
      checked_at: "2026-08-01",
    }));
    await expect(
      fetchLiveness({
        fetchFn: ok(feedOf(many)),
        sermons: sermons.map((s) => ({ ...s, scSuffixUrl: "aged-out" })),
        previous,
        checkedAt: "2026-08-07",
      }),
    ).rejects.toThrow(/dead fraction/i);
  });
});
