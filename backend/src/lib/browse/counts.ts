import type { SearchFilters } from "@ipp/shared";
import type { PrismaClient } from "@prisma/client";

/**
 * Facet counts adjusted to the filters already chosen.
 *
 * The "+ filtro" popover reads these. Offering the archive-wide totals there
 * would let someone pick "Gênesis (73)" on top of a preacher who never
 * preached it and land on an empty page -- the counts have to answer "how many
 * would be left", not "how many exist".
 *
 * Counted in memory over one query rather than six GROUP BYs: 456 rows of
 * facet keys is ~20 KB, it keeps the per-dimension exclusion rule below in one
 * readable place, and it makes the whole thing a pure function to test.
 */
export type CountRow = {
  artist: string;
  serviceType: string | null;
  seriesSlug: string | null;
  date: Date;
  scriptures: { bookSlug: string; chapter: number }[];
  topics: { topicSlug: string }[];
};

type FacetCounts = {
  pregadores: Record<string, number>;
  tipos: Record<string, number>;
  series: Record<string, number>;
  livros: Record<string, number>;
  temas: Record<string, number>;
  /** Keyed by year, because the popover filters by year, not by arbitrary range. */
  anos: Record<string, number>;
  /** How many sermons match every filter — what the result count will be. */
  total: number;
};

type Dimension = "pregadores" | "tipos" | "series" | "livros" | "temas" | "datas";

const inList = (list: string[] | undefined, value: string | null): boolean =>
  !list?.length || (value !== null && list.includes(value));

/**
 * One predicate per dimension, so the "all filters except this one" rule below
 * is a single `every` rather than six near-identical branches.
 *
 * The chapter belongs to the book predicate, not to a dimension of its own: it
 * only ever narrows a book, and dropping the book has to drop it too.
 */
const PREDICATES: Record<Dimension, (row: CountRow, f: SearchFilters) => boolean> = {
  pregadores: (r, f) => inList(f.pregadores, r.artist),
  tipos: (r, f) => inList(f.tipos, r.serviceType),
  series: (r, f) => inList(f.series, r.seriesSlug),
  livros: (r, f) =>
    !f.livros?.length ||
    r.scriptures.some(
      (s) => f.livros?.includes(s.bookSlug) && (!f.capitulo || s.chapter === f.capitulo),
    ),
  temas: (r, f) => !f.temas?.length || r.topics.some((t) => f.temas?.includes(t.topicSlug)),
  datas: (r, f) => {
    const iso = r.date.toISOString().slice(0, 10);
    return (!f.de || iso >= f.de) && (!f.ate || iso <= f.ate);
  },
};

const DIMENSIONS = Object.keys(PREDICATES) as Dimension[];

/** Matches every filter except `skip` — a dimension never constrains its own counts. */
const matches = (row: CountRow, f: SearchFilters, skip?: Dimension): boolean =>
  DIMENSIONS.every((d) => d === skip || PREDICATES[d](row, f));

const bump = (into: Record<string, number>, key: string | null): void => {
  if (key !== null) into[key] = (into[key] ?? 0) + 1;
};

export function countFacets(rows: CountRow[], filters: SearchFilters): FacetCounts {
  const counts: FacetCounts = {
    pregadores: {},
    tipos: {},
    series: {},
    livros: {},
    temas: {},
    anos: {},
    total: 0,
  };

  for (const row of rows) {
    if (matches(row, filters, "pregadores")) bump(counts.pregadores, row.artist);
    if (matches(row, filters, "tipos")) bump(counts.tipos, row.serviceType);
    if (matches(row, filters, "series")) bump(counts.series, row.seriesSlug);
    if (matches(row, filters, "livros")) {
      // A sermon spanning Gênesis 12-50 is 39 rows and one sermon.
      for (const book of new Set(row.scriptures.map((s) => s.bookSlug))) {
        bump(counts.livros, book);
      }
    }
    if (matches(row, filters, "temas")) {
      for (const topic of row.topics) bump(counts.temas, topic.topicSlug);
    }
    if (matches(row, filters, "datas")) {
      bump(counts.anos, String(row.date.getUTCFullYear()));
    }
    if (matches(row, filters)) counts.total += 1;
  }

  return counts;
}

export async function facetCounts(
  prisma: PrismaClient,
  filters: SearchFilters,
): Promise<FacetCounts> {
  const rows = await prisma.sermon.findMany({
    select: {
      artist: true,
      serviceType: true,
      seriesSlug: true,
      date: true,
      scriptures: { select: { bookSlug: true, chapter: true } },
      topics: { select: { topicSlug: true } },
    },
  });

  return countFacets(rows as CountRow[], filters);
}
