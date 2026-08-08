/**
 * Display helpers for sermon metadata. All output is pt-BR.
 *
 * The title and the date live in `@ipp/shared` because the server writes them
 * into the page before the browser does (see `backend/src/lib/seo/`). The
 * duration is ours alone: nothing prerendered shows it in a heading.
 */
export { formatDate, stripLeadingDate } from "@ipp/shared";

/** Drops a leading "0:" so "0:45:49" reads as "45:49". */
export function formatDuration(durationStr: string): string {
  return durationStr.replace(/^0:/, "");
}
