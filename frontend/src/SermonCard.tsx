import type { SearchResult } from "@ipp/shared";

/**
 * One search hit: the sermon it belongs to, the matched passage, and links to
 * play it. Audio always streams from SoundCloud/Spotify -- the 77 GB of
 * recordings is never hosted by this app.
 */

/** Splits text on the query's terms so matches can be marked. */
function highlight(text: string, query: string): (string | { match: string })[] {
  const terms = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 2)
    .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));

  if (terms.length === 0) return [text];

  // Accent-insensitive on the query side is handled by Postgres; here we only
  // mark literal matches, so a missed accent simply renders unhighlighted.
  const parts = text.split(new RegExp(`(${terms.join("|")})`, "gi"));
  return parts.map((part) =>
    terms.some((t) => new RegExp(`^${t}$`, "i").test(part)) ? { match: part } : part,
  );
}

function formatDate(iso: string): string {
  const [year, month, day] = iso.split("-");
  return `${day}/${month}/${year}`;
}

export function SermonCard({ result, query }: { result: SearchResult; query: string }) {
  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm transition hover:shadow-md dark:border-slate-700 dark:bg-slate-800">
      <header className="mb-2">
        <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
          {result.title}
        </h2>
        <p className="mt-0.5 text-sm text-slate-600 dark:text-slate-400">
          {result.artist} · {formatDate(result.date)}
          {result.durationStr ? ` · ${result.durationStr}` : ""}
        </p>
      </header>

      <p className="text-sm leading-relaxed text-slate-700 dark:text-slate-300">
        {highlight(result.content, query).map((part, i) =>
          typeof part === "string" ? (
            // biome-ignore lint/suspicious/noArrayIndexKey: split parts are positional
            <span key={i}>{part}</span>
          ) : (
            // biome-ignore lint/suspicious/noArrayIndexKey: split parts are positional
            <mark key={i} className="rounded bg-amber-200 px-0.5 dark:bg-amber-500/40">
              {part.match}
            </mark>
          ),
        )}
      </p>

      <footer className="mt-3 flex flex-wrap gap-3">
        {result.soundcloudUrl && (
          <a
            className="text-sm font-medium text-orange-600 hover:underline dark:text-orange-400"
            href={result.soundcloudUrl}
            target="_blank"
            rel="noreferrer"
          >
            Ouvir no SoundCloud
          </a>
        )}
        {result.spotifyUrl && (
          <a
            className="text-sm font-medium text-green-700 hover:underline dark:text-green-400"
            href={result.spotifyUrl}
            target="_blank"
            rel="noreferrer"
          >
            Ouvir no Spotify
          </a>
        )}
      </footer>
    </article>
  );
}
