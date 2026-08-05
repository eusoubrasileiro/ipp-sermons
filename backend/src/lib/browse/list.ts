import type { SearchFilters } from "@ipp/shared";
import type { PrismaClient } from "@prisma/client";

/**
 * Filtered sermon listings with no query text.
 *
 * Every index page links here. Kept apart from the facet counts because this
 * is the only browse query that grows with the corpus, and apart from
 * `search.ts` because it must keep working when the embedding API does not.
 */
export type BrowseSort = "data" | "biblia" | "serie";

/** A filtered listing with no query text — what every index page links to. */
export async function listSermons(
  prisma: PrismaClient,
  filters: SearchFilters,
  sort: BrowseSort = "data",
  page = 1,
  perPage = 20,
): Promise<{ total: number; sermons: Record<string, unknown>[] }> {
  const where: Record<string, unknown> = {};
  if (filters.pregadores?.length) where.artist = { in: filters.pregadores };
  if (filters.tipos?.length) where.serviceType = { in: filters.tipos };
  if (filters.series?.length) where.seriesSlug = { in: filters.series };
  if (filters.de || filters.ate) {
    where.date = {
      ...(filters.de ? { gte: new Date(`${filters.de}T00:00:00Z`) } : {}),
      ...(filters.ate ? { lte: new Date(`${filters.ate}T00:00:00Z`) } : {}),
    };
  }
  if (filters.livros?.length) {
    where.scriptures = {
      some: {
        bookSlug: { in: filters.livros },
        ...(filters.capitulo ? { chapter: filters.capitulo } : {}),
      },
    };
  }
  if (filters.temas?.length) {
    where.topics = { some: { topicSlug: { in: filters.temas } } };
  }

  // A course reads in order, and the lesson with no number is its introduction
  // -- nulls first, or "Introdução" lands after part 2.
  const orderBy =
    sort === "serie"
      ? [
          { seriesPart: { sort: "asc" as const, nulls: "first" as const } },
          { date: "asc" as const },
        ]
      : [{ date: "desc" as const }];

  const [total, sermons] = await Promise.all([
    prisma.sermon.count({ where }),
    prisma.sermon.findMany({
      where,
      orderBy,
      skip: (page - 1) * perPage,
      take: perPage,
      include: { scriptures: { include: { book: true } }, series: true },
    }),
  ]);

  return { total, sermons: sermons as unknown as Record<string, unknown>[] };
}
