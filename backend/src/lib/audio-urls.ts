/**
 * Rebuilds the public SoundCloud/Spotify links from the suffix columns the
 * corpus CSV stores.
 *
 * `sc_suffix_url` is yt-dlp's `webpage_url_basename` -- the track slug alone,
 * with no channel segment. A canonical SoundCloud track URL is
 * `https://soundcloud.com/<channel>/<slug>`, so the channel has to be added
 * back here or every link 404s.
 *
 * `sp_suffix_url` is the Spotify episode id straight from the Web API (22-char
 * base62); `https://open.spotify.com/episode/<id>` needs no show context.
 */

const SOUNDCLOUD_CHANNEL = "ipperegrinos";

/**
 * WORKAROUND -- Spotify links are suppressed for sermons preached before this
 * date, ratified by the product owner 2026-08-04.
 *
 * Roughly a third of the episode ids no longer resolve, and every dead one is
 * from 2019-2021 (sampled 40 ids through Spotify's oEmbed endpoint: 27 alive,
 * 13 dead, all pre-2022). The ids match what Spotify's own API returned when
 * the corpus was scraped and match an independent 2025 scrape, so this is not
 * an app bug and not a corrupt column -- the episodes were retired upstream,
 * most likely when the podcast changed hosts. SoundCloud covers 100% of the
 * corpus, so nothing becomes unreachable.
 *
 * Delete this constant and the guard in `spotifyUrl` if the old episode ids are
 * ever re-scraped.
 */
export const SPOTIFY_LINKS_ALIVE_FROM = "2022-01-01";

const SOUNDCLOUD_BASE = `https://soundcloud.com/${SOUNDCLOUD_CHANNEL}`;
const SPOTIFY_BASE = "https://open.spotify.com/episode";

function clean(suffix: string | null | undefined): string | null {
  if (!suffix) return null;
  // Tolerate a leading slash or a stray channel prefix so a future corpus
  // change in the suffix shape cannot silently produce a doubled path.
  const trimmed = suffix
    .trim()
    .replace(/^\/+/, "")
    .replace(/^ipperegrinos\//, "");
  return trimmed || null;
}

export function soundcloudUrl(suffix: string | null | undefined): string | null {
  const slug = clean(suffix);
  return slug ? `${SOUNDCLOUD_BASE}/${slug}` : null;
}

/**
 * `date` is the sermon's preaching date as `YYYY-MM-DD`. ISO dates compare
 * correctly as strings, so no Date parsing (and no timezone) is involved.
 * A missing or malformed date is treated as too old to trust.
 */
export function spotifyUrl(
  suffix: string | null | undefined,
  date: string | null | undefined,
): string | null {
  const id = clean(suffix);
  if (!id) return null;
  if (!date || date < SPOTIFY_LINKS_ALIVE_FROM) return null;
  return `${SPOTIFY_BASE}/${id}`;
}
