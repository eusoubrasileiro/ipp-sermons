import { highlight } from "../lib/highlight.ts";

/**
 * Text with the query's terms marked.
 *
 * One component rather than two, because the search excerpt and the reading
 * page were rendering byte-identical `<mark>` blocks. The matching itself is
 * accent-folded and prefix-based (see lib/highlight.ts), which is the part that
 * has to agree between the two: a term lit in the result and dark in the
 * transcript reads as the page having lost the passage.
 *
 * With an empty query the text is returned untouched, so callers do not have to
 * branch on whether a search is behind them.
 */
export function Highlighted({ text, query }: { text: string; query: string }) {
  if (!query) return <>{text}</>;

  return (
    <>
      {highlight(text, query).map((part, i) =>
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
    </>
  );
}
