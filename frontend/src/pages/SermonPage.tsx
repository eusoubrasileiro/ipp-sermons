import { useParams, useSearchParams } from "react-router-dom";
import { ErrorState } from "../components/States.tsx";
import { TranscriptBody } from "../components/TranscriptBody.tsx";
import { TranscriptHeader } from "../components/TranscriptHeader.tsx";
import { useTranscript } from "../lib/useTranscript.ts";

/**
 * Read a whole sermon.
 *
 * A route rather than a panel inside the results, for the reason every facet is
 * a route here: a member sends the link to someone in WhatsApp and it opens on
 * the passage that matters. A modal would also be the only one in this codebase
 * -- FilterPicker says why the project has none.
 *
 * `?q=` and `?trecho=` carry the search across, so arriving from a result lands
 * on the passage that matched with the terms still lit. Arriving from a browse
 * listing has neither, and simply opens at the top.
 */

/** Where "Voltar" goes: back to the search that produced this, or to the index. */
function backTarget(query: string): string {
  return query ? `/?q=${encodeURIComponent(query)}` : "/";
}

function chunkFrom(raw: string | null): number | null {
  if (raw === null) return null;
  const parsed = Number.parseInt(raw, 10);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

export function SermonPage() {
  const { id = "" } = useParams();
  const [params] = useSearchParams();
  const query = params.get("q") ?? "";
  const { status, transcript, error, retry } = useTranscript(id);

  if (status === "loading") {
    return (
      <div className="space-y-4" aria-busy="true">
        <div className="h-8 w-3/4 animate-pulse rounded bg-muted" />
        <div className="h-4 w-1/2 animate-pulse rounded bg-muted" />
        <div className="max-w-[68ch] space-y-3 pt-4">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-4 w-full animate-pulse rounded bg-muted" />
          ))}
        </div>
      </div>
    );
  }

  if (status === "error" || !transcript) {
    return <ErrorState message={error || "Sermão não encontrado."} onRetry={retry} />;
  }

  return (
    <article>
      <TranscriptHeader transcript={transcript} backTo={backTarget(query)} />
      <TranscriptBody
        text={transcript.text}
        query={query}
        chunkIndex={chunkFrom(params.get("trecho"))}
      />
    </article>
  );
}
