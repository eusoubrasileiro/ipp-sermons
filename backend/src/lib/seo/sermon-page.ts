import {
  formatDate,
  sermonTitle,
  stripLeadingDate,
  type TranscriptResponse,
  toParagraphs,
} from "@ipp/shared";
import { escapeHtml, type SeoPage, summarise } from "./html.ts";
import { SHELL_CLOSE, SHELL_OPEN } from "./layout.ts";

/**
 * A sermon as a document rather than as an empty shell.
 *
 * This is the whole point of the exercise: 560 sermons of roughly 6,400 words
 * each is 3.6 million words of exactly the long-tail Portuguese prose a search
 * engine rewards, and until now none of it was on a page a crawler could read.
 *
 * The description is the sermon's own opening words, not boilerplate. Every
 * sermon here shares a preacher, a church and a genre, so a templated
 * description would make 560 pages look like 560 copies of each other — which
 * is precisely what a search engine deduplicates away.
 */

/** Silent reading runs ~200 wpm; the number only has to be honest to the minute. */
const READING_WPM = 200;

/**
 * The disclaimer from `TranscriptHeader`, repeated here because a crawler's
 * snippet may be all anyone ever reads of this page. A preacher misquoted by a
 * transcription error is a real problem for the church that maintains this.
 */
const DISCLAIMER =
  "Transcrição automática do áudio. Pode conter erros — para citar, confira no original.";

function playLinks(transcript: TranscriptResponse): string {
  const links = [
    [transcript.soundcloudUrl, "Ouvir no SoundCloud"] as const,
    [transcript.spotifyUrl, "Ouvir no Spotify"] as const,
  ]
    .filter(([url]) => url !== null)
    .map(
      ([url, label]) =>
        `<a class="text-primary underline-offset-4 hover:underline" href="${escapeHtml(
          url as string,
        )}" rel="noopener">${label}</a>`,
    );

  return links.length === 0
    ? ""
    : `<p class="mt-3 flex flex-wrap gap-x-4 text-sm">${links.join("")}</p>`;
}

export function sermonPage(transcript: TranscriptResponse): SeoPage {
  const heading = stripLeadingDate(transcript.title);
  const minutes = Math.max(1, Math.round(transcript.words / READING_WPM));

  const meta = [
    escapeHtml(transcript.artist),
    `<time datetime="${escapeHtml(transcript.date)}">${escapeHtml(formatDate(transcript.date))}</time>`,
    transcript.durationStr ? escapeHtml(transcript.durationStr) : "",
    `${minutes} min de leitura`,
  ].filter(Boolean);

  const paragraphs = toParagraphs(transcript.text)
    .map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`)
    .join("");

  const body = [
    SHELL_OPEN,
    `<article>`,
    `<header>`,
    `<h1 class="font-display text-2xl font-bold leading-snug text-card-foreground sm:text-3xl">${escapeHtml(heading)}</h1>`,
    `<p class="mt-2 text-[0.8125rem] text-muted-foreground sm:text-sm">${meta.join(" · ")}</p>`,
    playLinks(transcript),
    `<p class="mt-4 rounded-md border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">${DISCLAIMER}</p>`,
    `</header>`,
    `<div class="mt-6 max-w-[68ch] space-y-4 text-[1.0625rem] leading-[1.75] text-card-foreground/95">${paragraphs}</div>`,
    `</article>`,
    `<p class="mt-8 text-sm"><a class="text-primary underline-offset-4 hover:underline" href="/">Buscar em todos os sermões</a></p>`,
    SHELL_CLOSE,
  ].join("");

  return {
    // The same builder the SPA uses when it re-titles the tab client-side.
    title: sermonTitle(transcript.title, transcript.artist),
    // A sermon with no usable text still needs a description that says what the
    // page is, or the search result is a bare URL.
    description:
      summarise(transcript.text) ||
      `Sermão pregado por ${transcript.artist} em ${formatDate(transcript.date)}.`,
    path: `/sermao/${encodeURIComponent(transcript.id)}`,
    ogType: "article",
    body,
  };
}
