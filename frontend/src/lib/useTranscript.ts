import type { TranscriptResponse } from "@ipp/shared";
import { useCallback, useEffect, useState } from "react";
import { fetchTranscript } from "../api.ts";

/**
 * Loads one sermon's full text.
 *
 * Simpler than `useSearch`: the id comes from the path and cannot change while
 * the page is mounted without React remounting it, so there is no stale-response
 * race to guard against. Retry re-runs the same call for the same reason it does
 * there — one description of how the request is made.
 */
type TranscriptStatus = "loading" | "done" | "error";

type TranscriptState = {
  status: TranscriptStatus;
  transcript: TranscriptResponse | null;
  error: string;
};

const LOADING: TranscriptState = { status: "loading", transcript: null, error: "" };

export function useTranscript(id: string): TranscriptState & { retry: () => void } {
  const [state, setState] = useState<TranscriptState>(LOADING);

  const run = useCallback(() => {
    setState(LOADING);
    fetchTranscript(id)
      .then((transcript) => setState({ status: "done", transcript, error: "" }))
      .catch((err: unknown) =>
        setState({
          status: "error",
          transcript: null,
          error: err instanceof Error ? err.message : "Não foi possível carregar a transcrição.",
        }),
      );
  }, [id]);

  useEffect(run, [run]);

  return { ...state, retry: run };
}
