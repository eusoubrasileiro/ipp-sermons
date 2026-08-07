/**
 * Which indexed sermons the corpus no longer offers.
 *
 * `index-corpus.ts` only ever upserted, so a row that left `metadata.csv` --
 * or that a loader filter began rejecting -- stayed in Postgres and stayed
 * searchable. Nine truncated sermons were the first result for their own
 * subject long after the CSV had stopped listing them. Adding rows and
 * removing them are the same job; only one of them was being done.
 */

/**
 * The most of the archive one run may delete.
 *
 * A pruner is a loaded gun pointed at production: it deletes exactly what the
 * loader declines to return, so any bug that makes `loadSermons` return too
 * little becomes a bug that empties the site. That is not hypothetical -- the
 * words/min filter this pruner exists to propagate rejected *every* row in its
 * first draft, because an absent column read as zero.
 *
 * 5% is roomy for real corrections (the nine were 1.8%) and nowhere near a
 * wipe. Same posture as `assertDeadFractionSane` in `podcast-feed.ts`.
 */
export const MAX_PRUNE_FRACTION = 0.05;

/**
 * Sermon ids to delete: indexed, but absent from the corpus.
 *
 * Ids the corpus has and the database does not are not this function's
 * business -- indexing adds those. Throws rather than returning a huge list,
 * because the caller's only sane response to an implausible plan is to stop.
 */
export function prunePlan(indexedIds: string[], corpusIds: string[]): string[] {
  if (indexedIds.length === 0) return [];

  if (corpusIds.length === 0) {
    throw new Error("corpus loaded 0 sermons — refusing to prune the whole index");
  }

  const keep = new Set(corpusIds);
  const remove = indexedIds.filter((id) => !keep.has(id));

  const fraction = remove.length / indexedIds.length;
  if (fraction > MAX_PRUNE_FRACTION) {
    throw new Error(
      `pruning would delete ${remove.length} of ${indexedIds.length} indexed sermons ` +
        `(${(fraction * 100).toFixed(1)}%, ceiling ${(MAX_PRUNE_FRACTION * 100).toFixed(1)}%) ` +
        `— refusing; check what the corpus loader is rejecting before re-running`,
    );
  }

  return remove;
}
