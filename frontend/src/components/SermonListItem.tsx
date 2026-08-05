import type { BrowseSermon } from "../api.ts";
import { formatDate, formatDuration, stripLeadingDate } from "../lib/format.ts";
import { PlayLinks } from "./PlayLinks.tsx";

const TYPE_LABELS: Record<string, string> = {
  culto: "Culto",
  ebd: "EBD",
  conferencia: "Conferência",
  congresso: "Congresso",
  diaconia: "Diaconia",
  confraria: "Confraria",
};

/**
 * A sermon in a browse listing.
 *
 * Distinct from SermonCard, which belongs to search and leads with the matched
 * excerpt. Here nothing matched -- the person chose a book or a course -- so
 * the passage and the series are what orient them instead.
 */
export function SermonListItem({ sermon }: { sermon: BrowseSermon }) {
  const title = sermon.displayTitle ?? stripLeadingDate(sermon.title);
  const passages = [...new Set(sermon.scriptures.map((s) => s.book.name))].slice(0, 2);

  const meta = [
    sermon.serviceType ? (TYPE_LABELS[sermon.serviceType] ?? sermon.serviceType) : null,
    formatDate(sermon.date.slice(0, 10)),
    sermon.artist,
    sermon.durationStr ? formatDuration(sermon.durationStr) : null,
  ].filter(Boolean);

  return (
    <article className="border-b border-border py-3">
      <p className="text-xs text-muted-foreground">{meta.join(" · ")}</p>
      <h3 className="mt-0.5 font-medium leading-snug">
        {sermon.seriesPart !== null ? (
          <span className="text-muted-foreground">{sermon.seriesPart}. </span>
        ) : null}
        {title}
      </h3>
      {passages.length > 0 || sermon.series ? (
        <p className="mt-0.5 text-xs text-muted-foreground">
          {[passages.join(", "), sermon.series?.name].filter(Boolean).join(" · ")}
        </p>
      ) : null}
      <div className="mt-2">
        <PlayLinks
          title={title}
          soundcloudUrl={
            sermon.scSuffixUrl ? `https://soundcloud.com/ipperegrinos/${sermon.scSuffixUrl}` : null
          }
          spotifyUrl={
            sermon.spSuffixUrl && sermon.date >= "2022-01-01"
              ? `https://open.spotify.com/episode/${sermon.spSuffixUrl}`
              : null
          }
        />
      </div>
    </article>
  );
}
