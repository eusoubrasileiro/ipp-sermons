import { slugify } from "./slugify.ts";

/**
 * Maps every raw series name back to the canonical slug it was folded into.
 *
 * `sermon_facets.csv` records what a title actually said -- "CFW 3",
 * "Atribututos de Deus". `series.csv` records what canonicalisation decided --
 * "CFW 3 — Do Decreto Eterno de Deus", "Atributos de Deus". The `variants`
 * column is the only bridge between them.
 *
 * Without it the load completes, reports success, and every renamed series
 * quietly loses all of its sermons: the page renders, empty. That already
 * happened once, which is why this is a tested function rather than four lines
 * inside the loader.
 */
export function buildVariantIndex(rows: Record<string, string>[]): Map<string, string> {
  const index = new Map<string, string>();

  for (const row of rows) {
    const slug = (row.slug ?? "").trim();
    if (!slug) continue;
    index.set(slug, slug);

    for (const variant of (row.variants ?? "").split("|")) {
      const name = variant.trim();
      if (name) index.set(slugify(name), slug);
    }
  }

  return index;
}

/** The canonical slug for a raw one, or null when nothing claims it. */
export function resolveSeries(index: Map<string, string>, rawSlug: string): string | null {
  const slug = rawSlug.trim();
  if (!slug) return null;
  return index.get(slug) ?? null;
}
