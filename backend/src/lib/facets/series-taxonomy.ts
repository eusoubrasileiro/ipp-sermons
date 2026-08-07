import type { NameCluster } from "./cluster.ts";
import { slugify } from "./slugify.ts";

/**
 * Assembles the canonical series table from the fuzzy clusters plus whatever
 * the LLM decided about them.
 *
 * Kept apart from the script that calls it because none of this should have to
 * be exercised by spending money: the merge resolution, the parent linkage and
 * the kind rules are pure functions of the model's answer, and every one of
 * them has a failure mode that shows up as a broken page rather than an error.
 */
export type SeriesDecision = {
  id: number;
  name: string;
  description: string;
  parent: string | null;
  merge_into: number | null;
};

export type SeriesRow = {
  slug: string;
  name: string;
  kind: string;
  parent_slug: string | null;
  parent_name: string | null;
  description: string;
  sermon_count: number;
  variants: string;
};

export const SERIES_COLUMNS = [
  "slug",
  "name",
  "kind",
  "parent_slug",
  "parent_name",
  "description",
  "sermon_count",
  "variants",
];

/** Folded name with any leading article dropped, for matching parent to child. */
export function lenientKey(name: string): string {
  return slugify(name).replace(/^(?:a|o|as|os)-/, "");
}

/** Deterministic; the model is not asked to guess what the title already says. */
export function kindOf(name: string): string {
  if (/^CFW\s*\d/i.test(name)) return "cfw";
  if (/confer[êe]ncia/i.test(name)) return "conferencia";
  if (/congresso/i.test(name)) return "congresso";
  if (/confraria/i.test(name)) return "confraria";
  if (/^diaconia$/i.test(name)) return "diaconia";
  return "ebd";
}

/**
 * The slug this cluster already publishes under, if it publishes under one.
 *
 * A slug is a URL; a name is a label. Deriving the first from the second means
 * every re-wording moves the page, and the model re-reads the Westminster
 * chapter names off example titles on every run -- "Da Escritura Sagrada" and
 * "Da Sagrada Escritura" are both reachable answers for the same chapter. The
 * `variants` column is what says the two are the same series, so once a slug is
 * published it stops following the name.
 */
function publishedSlug(cluster: NameCluster, committed?: Map<string, string>): string | null {
  if (!committed) return null;
  for (const member of cluster.members) {
    const prior = committed.get(slugify(member));
    if (prior) return prior;
  }
  return null;
}

/**
 * @param committed variant slug -> published series slug, from the committed
 *   series.csv via `buildVariantIndex`. Omit on a first run.
 */
export function buildSeriesRows(
  clusters: NameCluster[],
  decisions: SeriesDecision[],
  committed?: Map<string, string>,
): SeriesRow[] {
  const byId = new Map(decisions.map((d) => [d.id, d]));

  /** Follows merge_into to its end, guarding against a cycle the model invents. */
  const resolve = (id: number, hops = 0): number => {
    const target = byId.get(id)?.merge_into;
    if (target === null || target === undefined || !byId.has(target) || hops > 8) return id;
    return resolve(target, hops + 1);
  };

  const rows = new Map<string, SeriesRow>();
  for (let id = 0; id < clusters.length; id++) {
    const cluster = clusters[id] as NameCluster;
    const root = resolve(id);
    const decided = byId.get(root);
    const name = decided?.name?.trim() || (clusters[root] as NameCluster).provisional;
    const slug =
      publishedSlug(cluster, committed) ??
      publishedSlug(clusters[root] as NameCluster, committed) ??
      slugify(name);

    const existing = rows.get(slug);
    if (existing) {
      existing.sermon_count += cluster.count;
      existing.variants = `${existing.variants}|${cluster.members.join("|")}`;
      continue;
    }

    rows.set(slug, {
      slug,
      name,
      // The adjudicated name usually carries the kind ("CFW 3 — ..."), but when
      // it does not, the raw cluster name still might.
      kind: kindOf(name) === "ebd" ? kindOf(cluster.provisional) : kindOf(name),
      parent_slug: decided?.parent ? slugify(decided.parent) : null,
      parent_name: decided?.parent ?? null,
      description: decided?.description ?? "",
      sermon_count: cluster.count,
      variants: cluster.members.join("|"),
    });
  }

  // Point every parent reference at a row that actually exists.
  //
  // The model names the parent in prose, so whether it writes "Confissão de Fé
  // de Westminster" or "A Confissão de Fé de Westminster" is a coin flip -- and
  // that one article is enough to orphan all twelve Westminster chapters from
  // the intro lesson that heads them. It happened. Match leniently rather than
  // trust the model to be consistent with itself.
  const byLenient = new Map([...rows.values()].map((r) => [lenientKey(r.name), r.slug]));
  for (const row of rows.values()) {
    if (!row.parent_slug) continue;
    const resolved = byLenient.get(lenientKey(row.parent_name ?? row.parent_slug));
    if (resolved && resolved !== row.slug) row.parent_slug = resolved;
    // A parent nothing resolves to is still a useful grouping label, so it is
    // left in place rather than dropped.
  }

  // A series that other series point at heads a course, whatever its own name
  // looks like: the Westminster Confession is both a real intro lesson and the
  // parent of the twelve chapters.
  const parents = new Set([...rows.values()].map((r) => r.parent_slug).filter(Boolean));
  for (const row of rows.values()) {
    if (parents.has(row.slug)) row.kind = "cfw";
  }

  return [...rows.values()].sort(
    (a, b) => b.sermon_count - a.sermon_count || a.name.localeCompare(b.name),
  );
}

/**
 * Committed series slugs the proposed taxonomy would remove.
 *
 * `canonicalize:series` rewrites this file in full from a model answer that is
 * not deterministic, and merging two names into one is the job it exists to do.
 * Growing the taxonomy is harmless. Losing a slug is not: a slug already in a
 * commit is a `/series/<slug>` URL that is live, and the pass has no way to
 * know which of the names it is folding together somebody has linked to.
 *
 * git draws the line. A slug in the last commit is protected; a slug that only
 * exists in this run is still the model's to rearrange.
 */
export function retiredSlugs(committed: string[], proposed: string[]): string[] {
  const kept = new Set(proposed);
  return committed.filter((slug) => !kept.has(slug));
}
