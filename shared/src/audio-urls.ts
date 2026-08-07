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
 * `alive` is the sermon's `spotify_alive` column, derived from the podcast feed
 * in `data/facets/spotify_episodes.csv` -- see `podcast-feed.ts` for why feed
 * membership, and not the sermon's date, decides this.
 *
 * Unknown liveness suppresses the link. A sermon indexed before the check has
 * run has no answer yet, and a dead play button is worse than none.
 */
export function spotifyUrl(
  suffix: string | null | undefined,
  alive: boolean | null | undefined,
): string | null {
  const id = clean(suffix);
  if (!id || !alive) return null;
  return `${SPOTIFY_BASE}/${id}`;
}
