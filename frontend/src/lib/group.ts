import type { SearchResult } from "@ipp/shared";

/**
 * The API ranks transcript *chunks*, so one sermon can occupy several slots of
 * a ten-result page. For someone looking for a sermon to listen to that reads
 * as duplicates and pushes other sermons off the screen -- so the card is the
 * sermon, and its extra matching passages hang off it.
 *
 * Order is by best-ranked chunk, which is the order the API already returns.
 */

export type SermonGroup = {
  /** Best-ranked hit for this sermon; carries the metadata and the audio links. */
  top: SearchResult;
  /** Further matching passages, best first. */
  more: SearchResult[];
};

export function groupBySermon(results: SearchResult[]): SermonGroup[] {
  const groups = new Map<string, SermonGroup>();
  for (const r of results) {
    const existing = groups.get(r.id);
    if (existing) existing.more.push(r);
    else groups.set(r.id, { top: r, more: [] });
  }
  return [...groups.values()];
}
