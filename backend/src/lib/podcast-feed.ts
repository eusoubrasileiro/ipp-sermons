/**
 * Reads the church's podcast RSS feed to decide which Spotify episodes are
 * still reachable.
 *
 * The feed is served by SoundCloud, and every platform the church publishes to
 * — Spotify, Apple Podcasts, Deezer, YouTube Music — ingests this one URL.
 * **It is capped at 500 items.** When a sermon falls out of that window the
 * aggregators delist its episode, and `open.spotify.com/episode/<id>` starts
 * answering 404 even though the id itself was never wrong.
 *
 * That makes feed membership the ground truth for "does this Spotify link
 * work". Probing Spotify directly would be more direct but it rate-limits
 * hard (429 on a few hundred ids), and one RSS request answers for the whole
 * corpus at once.
 *
 * The window *rolls*: roughly 50-100 new sermons a year push the oldest off,
 * so this answer expires. That is why the derived file carries `checked_at`
 * and why the check is re-runnable on its own, not only when new sermons land.
 */

/** Discovered from the Apple Podcasts id on https://www.ipperegrinos.com/gravacoes. */
const PODCAST_FEED_URL = "https://feeds.soundcloud.com/users/soundcloud:users:695742830/sounds.rss";

/**
 * A healthy feed carries 500 items. Anything under this is a truncated or
 * failed response, not a shrunken archive — treat it as a failure rather than
 * writing several hundred episodes off as dead.
 */
export const MIN_FEED_ITEMS = 100;

/**
 * How far the dead fraction may climb in one run before we assume the feed
 * lied. A rolling window retires a few percent between corpus updates; a jump
 * of this size means the response was short, not that the archive vanished.
 */
const MAX_DEAD_FRACTION_JUMP = 0.15;

/** Only track URLs — not the channel `<link>`, the stream enclosure or the artwork. */
const TRACK_LINK = /https:\/\/soundcloud\.com\/ipperegrinos\/([^<"\s]+)/g;

export type LivenessRow = {
  sermon_id: string;
  spotify_id: string;
  alive: boolean;
  checked_at: string;
};

/** The subset of a corpus record this needs, so the corpus type stays out of here. */
type FeedSermon = {
  id: string;
  scSuffixUrl: string | null;
  spSuffixUrl: string | null;
};

/**
 * Track slugs in feed order, de-duplicated.
 *
 * Matched by regex rather than parsed as XML: the one shape we care about is
 * unambiguous, and it keeps an XML dependency out of the runtime image.
 */
export function parseFeedSlugs(xml: string): string[] {
  const seen = new Set<string>();
  for (const [, slug] of xml.matchAll(TRACK_LINK)) {
    if (slug) seen.add(slug);
  }
  return [...seen];
}

/** Strips the leading slash / channel prefix `audio-urls.ts` also tolerates. */
const slugOf = (suffix: string | null): string =>
  (suffix ?? "")
    .trim()
    .replace(/^\/+/, "")
    .replace(/^ipperegrinos\//, "");

export function assertFeedSane(slugs: string[]): void {
  if (slugs.length === 0) {
    throw new Error(`Podcast feed returned 0 items — refusing to mark the archive dead`);
  }
  if (slugs.length < MIN_FEED_ITEMS) {
    throw new Error(
      `Podcast feed returned too few items (${slugs.length} < ${MIN_FEED_ITEMS}) — refusing to write`,
    );
  }
}

/**
 * One row per sermon that *has* a Spotify episode id.
 *
 * Sermons that never had one are left out entirely: "no episode" and "episode
 * retired" are different facts, and only the second is this file's business.
 */
export function livenessRows(
  sermons: FeedSermon[],
  liveSlugs: Set<string>,
  checkedAt: string,
): LivenessRow[] {
  const rows: LivenessRow[] = [];
  for (const s of sermons) {
    const spotifyId = (s.spSuffixUrl ?? "").trim();
    if (!spotifyId) continue;
    rows.push({
      sermon_id: s.id,
      spotify_id: spotifyId,
      alive: liveSlugs.has(slugOf(s.scSuffixUrl)),
      checked_at: checkedAt,
    });
  }
  return rows;
}

/** Reads back what a previous run wrote, so the dead-fraction guard has a baseline. */
export function parseLivenessCsv(rows: Record<string, string>[]): LivenessRow[] {
  return rows.map((r) => ({
    sermon_id: (r.sermon_id ?? "").trim(),
    spotify_id: (r.spotify_id ?? "").trim(),
    alive: (r.alive ?? "").trim() === "true",
    checked_at: (r.checked_at ?? "").trim(),
  }));
}

/**
 * The whole decision, with the network injected.
 *
 * Lives here rather than in the script so the failure paths that matter -- a
 * feed that 500s, a feed that comes back short, a run that would blank most of
 * the archive -- are reachable from a test without a real HTTP call.
 */
export async function fetchLiveness(deps: {
  fetchFn: typeof fetch;
  sermons: FeedSermon[];
  previous: LivenessRow[];
  checkedAt: string;
}): Promise<{ rows: LivenessRow[]; feedItems: number }> {
  const response = await deps.fetchFn(PODCAST_FEED_URL);
  if (!response.ok) {
    throw new Error(`Podcast feed returned HTTP ${response.status} — refusing to write`);
  }
  const slugs = parseFeedSlugs(await response.text());
  assertFeedSane(slugs);

  const rows = livenessRows(deps.sermons, new Set(slugs), deps.checkedAt);
  assertDeadFractionSane(deps.previous, rows);
  return { rows, feedItems: slugs.length };
}

const deadFraction = (rows: LivenessRow[]): number =>
  rows.length === 0 ? 0 : rows.filter((r) => !r.alive).length / rows.length;

/**
 * Guards against the failure the feed itself cannot signal: a response that
 * parses, clears `assertFeedSane`, and is still missing most of the archive.
 * Episodes coming *back* is never suspicious, so this is one-sided.
 */
export function assertDeadFractionSane(previous: LivenessRow[], next: LivenessRow[]): void {
  if (previous.length === 0) return;
  const jump = deadFraction(next) - deadFraction(previous);
  if (jump > MAX_DEAD_FRACTION_JUMP) {
    throw new Error(
      `dead fraction jumped ${(jump * 100).toFixed(1)}pp in one run ` +
        `(${(deadFraction(previous) * 100).toFixed(1)}% → ${(deadFraction(next) * 100).toFixed(1)}%) ` +
        `— refusing to write; re-run to confirm the feed is healthy`,
    );
  }
}
