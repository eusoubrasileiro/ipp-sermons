import { SoundCloudIcon, SpotifyIcon } from "./BrandIcons.tsx";

/**
 * The point of the whole product: a result nobody can play is worthless, so the
 * links are full-size buttons in the platform's own colour rather than text
 * links. Both sit on a 44px touch target and carry the sermon title in their
 * accessible name, since "Spotify" alone is meaningless out of context in a
 * screen-reader's link list.
 */

// Stretches to fill the row on a phone (thumb-sized targets); settles to its
// natural width on wider screens, where a full-bleed orange bar shouts.
const BUTTON =
  "inline-flex min-h-[44px] flex-1 items-center justify-center gap-2 rounded-md px-5 py-2.5 text-sm font-semibold shadow-sm transition hover:brightness-110 active:brightness-95 sm:min-w-[11rem] sm:flex-none";

export function PlayLinks({
  title,
  soundcloudUrl,
  spotifyUrl,
}: {
  title: string;
  soundcloudUrl: string | null;
  spotifyUrl: string | null;
}) {
  if (!soundcloudUrl && !spotifyUrl) {
    return (
      <p className="mt-4 text-sm text-muted-foreground">Áudio indisponível para este sermão.</p>
    );
  }

  return (
    <div className="mt-4">
      <p className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        Ouvir
      </p>
      <div className="flex flex-wrap gap-2">
        {soundcloudUrl && (
          <a
            href={soundcloudUrl}
            target="_blank"
            rel="noreferrer"
            aria-label={`Ouvir "${title}" no SoundCloud`}
            className={`${BUTTON} bg-soundcloud text-soundcloud-foreground`}
          >
            <SoundCloudIcon className="h-5 w-5 shrink-0" />
            SoundCloud
          </a>
        )}
        {spotifyUrl && (
          <a
            href={spotifyUrl}
            target="_blank"
            rel="noreferrer"
            aria-label={`Ouvir "${title}" no Spotify`}
            className={`${BUTTON} bg-spotify text-spotify-foreground`}
          >
            <SpotifyIcon className="h-5 w-5 shrink-0" />
            Spotify
          </a>
        )}
      </div>
    </div>
  );
}
