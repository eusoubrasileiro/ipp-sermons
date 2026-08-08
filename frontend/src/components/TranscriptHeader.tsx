import type { TranscriptResponse } from "@ipp/shared";
import { Link } from "react-router-dom";
import { formatDate, formatDuration, stripLeadingDate } from "../lib/format.ts";
import { PlayLinks } from "./PlayLinks.tsx";

/**
 * What the reader needs before the first paragraph: which sermon this is, how
 * to hear it instead, roughly how long the read is, and that a machine wrote
 * the text.
 *
 * The disclaimer is not decoration. This is a church's archive, and a preacher
 * misquoted by a transcription error is a real problem for the people who
 * maintain it -- so it sits above the text, not in a footer.
 */

/** Silent reading runs ~200 wpm; the number only has to be honest to the minute. */
const READING_WPM = 200;

export function TranscriptHeader({
  transcript,
  backTo,
}: {
  transcript: TranscriptResponse;
  backTo: string;
}) {
  const title = stripLeadingDate(transcript.title);
  const minutes = Math.max(1, Math.round(transcript.words / READING_WPM));

  return (
    <header>
      <Link
        to={backTo}
        className="inline-flex min-h-11 items-center rounded text-sm font-medium text-primary underline-offset-4 hover:underline"
      >
        ← Voltar
      </Link>

      <h1 className="mt-1 font-display text-2xl font-bold leading-snug text-card-foreground sm:text-3xl">
        {title}
      </h1>

      <p className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[0.8125rem] text-muted-foreground sm:text-sm">
        <span className="font-medium text-foreground/80">{transcript.artist}</span>
        <span aria-hidden="true">·</span>
        <span>{formatDate(transcript.date)}</span>
        {transcript.durationStr && (
          <>
            <span aria-hidden="true">·</span>
            <span>{formatDuration(transcript.durationStr)}</span>
          </>
        )}
        <span aria-hidden="true">·</span>
        <span>{minutes} min de leitura</span>
      </p>

      <PlayLinks
        title={title}
        soundcloudUrl={transcript.soundcloudUrl}
        spotifyUrl={transcript.spotifyUrl}
      />

      <p className="mt-4 rounded-md border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
        Transcrição automática do áudio. Pode conter erros — para citar, confira no original.
      </p>
    </header>
  );
}
