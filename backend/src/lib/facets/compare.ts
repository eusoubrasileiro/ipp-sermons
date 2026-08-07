/**
 * Whether two labelling configurations disagree, and by how much.
 *
 * Two questions have to be answered before anything in the corpus is
 * reclassified: does reading the whole transcript beat the three-window sample,
 * and does a cheaper model hold up in Portuguese. Neither is settled by
 * argument, and neither is worth a human reading forty sermons — so the bench
 * labels all of them several ways and hands back only the rows where the
 * configurations actually differ.
 *
 * This measures *consistency*, not correctness: it says two configurations
 * disagree, never which one is right. That is deliberate. Consistency is the
 * risk that matters for a model swap — a labeller that is merely different
 * makes recent sermons incomparable with the rest of the corpus — and it costs
 * no human attention to measure. Correctness needs a person, and lives in
 * `docs/facet-quality.md` until there is time for it.
 *
 * Order never carries meaning. `label-topics` returns topics most-central
 * first, but two configurations picking the same three in a different order
 * agree about the sermon.
 */
export type Labelled = {
  sermonId: string;
  title: string;
  /** Topic slugs per configuration id; a configuration that failed is absent. */
  byConfig: Record<string, string[]>;
};

type Agreement = {
  config: string;
  /** Rows whose topic set is identical to the baseline's. */
  exact: number;
  /** Mean Jaccard against the baseline, so near-misses score above misses. */
  jaccard: number;
  total: number;
};

const asSet = (topics: string[] | undefined): Set<string> => new Set(topics ?? []);

export function sameTopics(a: string[], b: string[]): boolean {
  const left = asSet(a);
  const right = asSet(b);
  return left.size === right.size && [...left].every((t) => right.has(t));
}

export function jaccard(a: string[], b: string[]): number {
  const left = asSet(a);
  const right = asSet(b);
  // Neither answering is agreement. Left as 0/0 it would be NaN, and averaging
  // it would poison the whole column.
  if (left.size === 0 && right.size === 0) return 1;

  const shared = [...left].filter((t) => right.has(t)).length;
  return shared / (left.size + right.size - shared);
}

/** The rows worth a human's attention: the ones where the configurations differ. */
export function divergent(rows: Labelled[], configs: string[]): Labelled[] {
  return rows.filter((row) => {
    const [first, ...rest] = configs;
    if (first === undefined) return false;
    // A configuration that produced nothing counts as disagreeing: a missing
    // answer is exactly what somebody should look at.
    return rest.some((c) => !sameTopics(row.byConfig[first] ?? [], row.byConfig[c] ?? []));
  });
}

/** How each configuration compares against the one already in the corpus. */
export function agreementWith(rows: Labelled[], baseline: string, configs: string[]): Agreement[] {
  return configs
    .filter((c) => c !== baseline)
    .map((config) => {
      const scores = rows.map((row) => ({
        exact: sameTopics(row.byConfig[baseline] ?? [], row.byConfig[config] ?? []),
        jaccard: jaccard(row.byConfig[baseline] ?? [], row.byConfig[config] ?? []),
      }));

      const total = scores.length;
      return {
        config,
        exact: scores.filter((s) => s.exact).length,
        jaccard: total === 0 ? 0 : scores.reduce((sum, s) => sum + s.jaccard, 0) / total,
        total,
      };
    });
}
