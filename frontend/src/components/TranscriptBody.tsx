import { useEffect, useMemo, useRef } from "react";
import { CHUNK_STRIDE, paragraphAtWord, toParagraphs } from "../lib/paragraphs.ts";
import { Highlighted } from "./Highlighted.tsx";

/**
 * The sermon itself, set for reading.
 *
 * Everything is in the DOM at once. A transcript runs 6,400 words and
 * virtualising it would be the obvious optimisation, but it would also break
 * the browser's own Ctrl+F — which is the only navigation a text with no
 * headings has. 40 KB of text is nothing next to that.
 *
 * Paragraphs are invented (see lib/paragraphs.ts); the corpus has no line
 * breaks at all. The anchor is not: it comes from the chunk that actually
 * matched the search.
 */

/** ~68 characters a line, the width prose stops being tiring to read at. */
const COLUMN = "max-w-[68ch]";

function Paragraph({ text, query }: { text: string; query: string }) {
  return (
    <p>
      <Highlighted text={text} query={query} />
    </p>
  );
}

export function TranscriptBody({
  text,
  query,
  chunkIndex,
}: {
  text: string;
  query: string;
  chunkIndex: number | null;
}) {
  const paragraphs = useMemo(() => toParagraphs(text), [text]);
  const anchor = useMemo(
    () => (chunkIndex === null ? null : paragraphAtWord(paragraphs, chunkIndex * CHUNK_STRIDE)),
    [paragraphs, chunkIndex],
  );

  const anchorRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // Centred rather than at the top: the passage means little without the
    // sentences that led into it. `instant` because a smooth scroll through
    // 6,000 words takes seconds and reads as a bug.
    //
    // Feature-detected because scrollIntoView is not universal -- jsdom has no
    // implementation at all, and neither do some embedded webviews. Not being
    // scrolled to the passage is a worse read; a thrown TypeError is a blank page.
    const anchorEl = anchorRef.current;
    if (typeof anchorEl?.scrollIntoView === "function") {
      anchorEl.scrollIntoView({ block: "center", behavior: "instant" });
    }
  }, []);

  if (paragraphs.length === 0) {
    return (
      <p className="mt-6 text-muted-foreground">Este sermão não tem transcrição disponível.</p>
    );
  }

  return (
    <div
      className={`${COLUMN} mt-6 space-y-4 text-[1.0625rem] leading-[1.75] text-card-foreground/95 sm:text-lg sm:leading-[1.8]`}
    >
      {paragraphs.map((paragraph, i) =>
        i === anchor ? (
          <div
            // biome-ignore lint/suspicious/noArrayIndexKey: paragraphs are positional slices of one string
            key={i}
            ref={anchorRef}
            className="-ml-4 border-l-2 border-gold-rule pl-4"
          >
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Trecho que respondeu à sua busca
            </p>
            <Paragraph text={paragraph} query={query} />
          </div>
        ) : (
          // Unhighlighted on purpose. Lighting every occurrence of the query
          // across 7,000 words speckles the page -- "dever" and "homem" hit 132
          // times in one sermon, and prefix matching splits "dever|es" mid-word.
          // That is fine in a 200-word excerpt and unreadable at this length, so
          // the marks stay where they answer something: the matched passage.
          // Ctrl+F still finds the rest, which is why the text is all in the DOM.
          // biome-ignore lint/suspicious/noArrayIndexKey: paragraphs are positional slices of one string
          <Paragraph key={i} text={paragraph} query="" />
        ),
      )}
    </div>
  );
}
