import { useId, useState } from "react";
import { formatDate, formatDuration, stripLeadingDate } from "../lib/format.ts";
import type { SermonGroup } from "../lib/group.ts";
import { highlight, snippet } from "../lib/highlight.ts";
import { Card } from "./Card.tsx";
import { PlayLinks } from "./PlayLinks.tsx";

/**
 * One sermon: what it is, the passage that matched, and how to play it. Audio
 * always streams from SoundCloud/Spotify -- the recordings are never hosted by
 * this app.
 */

function Excerpt({ text, query, clamp }: { text: string; query: string; clamp: boolean }) {
  // Collapsed, show the window around the match rather than the chunk's opening
  // words; expanded, show the chunk as it was transcribed.
  const shown = clamp ? snippet(text, query) : text;
  return (
    <p
      className={`font-display text-[1.02rem] italic leading-relaxed text-card-foreground/90 ${clamp ? "line-clamp-4" : ""}`}
    >
      {highlight(shown, query).map((part, i) =>
        part.match ? (
          <mark
            // biome-ignore lint/suspicious/noArrayIndexKey: parts are positional slices of one string
            key={i}
            className="rounded bg-highlight px-0.5 font-medium text-highlight-foreground"
          >
            {part.text}
          </mark>
        ) : (
          // biome-ignore lint/suspicious/noArrayIndexKey: parts are positional slices of one string
          <span key={i}>{part.text}</span>
        ),
      )}
    </p>
  );
}

export function SermonCard({ group, query }: { group: SermonGroup; query: string }) {
  const [expanded, setExpanded] = useState(false);
  const passagesId = useId();
  const { top, more } = group;
  const title = stripLeadingDate(top.title);

  return (
    <Card as="article">
      <h2 className="font-display text-xl font-bold leading-snug text-card-foreground sm:text-2xl">
        {title}
      </h2>

      <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[0.8125rem] text-muted-foreground sm:text-sm">
        <span className="font-medium text-foreground/80">{top.artist}</span>
        <span aria-hidden="true">·</span>
        <span>{formatDate(top.date)}</span>
        {top.durationStr && (
          <>
            <span aria-hidden="true">·</span>
            <span>{formatDuration(top.durationStr)}</span>
          </>
        )}
      </p>

      {/* The church sets scripture in a gold italic serif; this echoes it, at a
          gold dark enough to pass contrast on body-sized text. */}
      <blockquote id={passagesId} className="mt-3 border-l-2 border-gold-rule pl-4">
        <Excerpt text={top.content} query={query} clamp={!expanded} />

        {expanded &&
          more.map((m) => (
            <div key={m.chunkIndex} className="mt-3 border-t border-border/60 pt-3">
              <Excerpt text={m.content} query={query} clamp={false} />
            </div>
          ))}
      </blockquote>

      <button
        type="button"
        aria-expanded={expanded}
        aria-controls={passagesId}
        onClick={() => setExpanded((v) => !v)}
        className="mt-2 rounded text-sm font-medium text-primary underline-offset-4 hover:underline"
      >
        {expanded
          ? "Mostrar menos"
          : more.length > 0
            ? `Ver trecho completo e mais ${more.length} ${more.length === 1 ? "passagem" : "passagens"}`
            : "Ver trecho completo"}
      </button>

      <PlayLinks title={title} soundcloudUrl={top.soundcloudUrl} spotifyUrl={top.spotifyUrl} />
    </Card>
  );
}
