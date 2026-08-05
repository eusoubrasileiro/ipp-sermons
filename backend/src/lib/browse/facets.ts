import type { PrismaClient } from "@prisma/client";
import { slugify } from "../facets/slugify.ts";

/**
 * The browse side of the site: index pages with counts, and filtered sermon
 * listings with no query text at all.
 *
 * Deliberately separate from `search.ts`. Search ranks by relevance and needs
 * an embedding, a reranker and a paid API call; browsing is an ordinary
 * indexed query over 456 rows and must keep working when the model does not.
 * Sharing a code path would make an outage of the one an outage of the other.
 *
 * Counts come from GROUP BY over the whole table rather than from
 * hybrid_search(): a facet index shows how much exists, not how much matched.
 */
type BookFacet = {
  slug: string;
  nome: string;
  testamento: string;
  ordem: number;
  total: number;
  capitulos: { numero: number; total: number }[];
};

type SeriesFacet = {
  slug: string;
  nome: string;
  kind: string;
  paiSlug: string | null;
  descricao: string;
  total: number;
};

/**
 * `artist` is the raw column and the only thing a filter may be built from:
 * `titulo` is "Outros" for the one preacher with no honorific, so joining the
 * two back together would produce a name that matches nothing.
 */
type PreacherFacet = {
  slug: string;
  artist: string;
  nome: string;
  titulo: string;
  total: number;
};
type DateFacet = { ano: number; total: number; meses: { mes: number; total: number }[] };
type TypeFacet = { slug: string; total: number };
type TopicFacet = {
  slug: string;
  nome: string;
  grupoSlug: string;
  grupoNome: string;
  total: number;
};

export type FacetTree = {
  livros: BookFacet[];
  series: SeriesFacet[];
  pregadores: PreacherFacet[];
  datas: DateFacet[];
  tipos: TypeFacet[];
  temas: TopicFacet[];
};

const num = (v: unknown): number => Number(v ?? 0);

/**
 * Splits "Reverendo Bruno Melo" into its honorific and the person.
 *
 * The honorific groups the index page; the name identifies the preacher. We
 * have no photographs, so the initials of the name become the avatar.
 */
export function splitPreacher(artist: string): { titulo: string; nome: string } {
  const match = artist.match(
    /^(Reverendo|Pastor|Presb[ií]tero|Semin[aá]rista|Seminarista|Di[aá]cono|Mission[aá]rio)\s+(.*)$/i,
  );
  if (!match?.[1] || !match[2]) return { titulo: "Outros", nome: artist };
  return { titulo: match[1], nome: match[2] };
}

async function bookFacets(prisma: PrismaClient): Promise<BookFacet[]> {
  const rows = await prisma.$queryRaw<
    {
      slug: string;
      nome: string;
      testamento: string;
      ordem: number;
      chapter: number;
      total: bigint;
    }[]
  >`
    SELECT b.slug, b.name AS nome, b.testament AS testamento, b.canon_order AS ordem,
           ss.chapter, count(DISTINCT ss.sermon_id) AS total
    FROM bible_books b
    JOIN sermon_scriptures ss ON ss.book_slug = b.slug
    GROUP BY b.slug, b.name, b.testament, b.canon_order, ss.chapter
    ORDER BY b.canon_order, ss.chapter`;

  const books = new Map<string, BookFacet>();
  for (const r of rows) {
    const book = books.get(r.slug) ?? {
      slug: r.slug,
      nome: r.nome,
      testamento: r.testamento,
      ordem: num(r.ordem),
      total: 0,
      capitulos: [],
    };
    // Chapter 0 means the title named the book without a chapter; it counts
    // toward the book but is not a chapter anyone can browse to.
    if (num(r.chapter) > 0) {
      book.capitulos.push({ numero: num(r.chapter), total: num(r.total) });
    }
    books.set(r.slug, book);
  }

  // The book total is distinct sermons, not the sum of its chapters: one
  // sermon on "Gênesis 12-50" contributes 39 chapter rows and one sermon.
  const totals = await prisma.$queryRaw<{ slug: string; total: bigint }[]>`
    SELECT book_slug AS slug, count(DISTINCT sermon_id) AS total
    FROM sermon_scriptures GROUP BY book_slug`;
  for (const t of totals) {
    const book = books.get(t.slug);
    if (book) book.total = num(t.total);
  }

  return [...books.values()].sort((a, b) => a.ordem - b.ordem);
}

async function seriesFacets(prisma: PrismaClient): Promise<SeriesFacet[]> {
  const rows = await prisma.$queryRaw<
    {
      slug: string;
      nome: string;
      kind: string;
      pai: string | null;
      descricao: string;
      total: bigint;
    }[]
  >`
    SELECT se.slug, se.name AS nome, se.kind, se.parent_slug AS pai, se.description AS descricao,
           count(sm.id) AS total
    FROM series se
    LEFT JOIN sermons sm ON sm.series_slug = se.slug
    GROUP BY se.slug, se.name, se.kind, se.parent_slug, se.description
    ORDER BY se.name`;

  return rows.map((r) => ({
    slug: r.slug,
    nome: r.nome,
    kind: r.kind,
    paiSlug: r.pai,
    descricao: r.descricao,
    total: num(r.total),
  }));
}

async function preacherFacets(prisma: PrismaClient): Promise<PreacherFacet[]> {
  const rows = await prisma.$queryRaw<{ artist: string; total: bigint }[]>`
    SELECT artist, count(*) AS total FROM sermons GROUP BY artist ORDER BY count(*) DESC`;

  return rows.map((r) => ({
    slug: slugify(r.artist),
    artist: r.artist,
    total: num(r.total),
    ...splitPreacher(r.artist),
  }));
}

async function dateFacets(prisma: PrismaClient): Promise<DateFacet[]> {
  const rows = await prisma.$queryRaw<{ ano: number; mes: number; total: bigint }[]>`
    SELECT EXTRACT(YEAR FROM date)::int AS ano, EXTRACT(MONTH FROM date)::int AS mes,
           count(*) AS total
    FROM sermons GROUP BY 1, 2 ORDER BY 1 DESC, 2 DESC`;

  const years = new Map<number, DateFacet>();
  for (const r of rows) {
    const year = years.get(num(r.ano)) ?? { ano: num(r.ano), total: 0, meses: [] };
    year.meses.push({ mes: num(r.mes), total: num(r.total) });
    year.total += num(r.total);
    years.set(num(r.ano), year);
  }
  // Newest first: "the sermon from last Sunday" is the commonest real request.
  return [...years.values()].sort((a, b) => b.ano - a.ano);
}

async function typeFacets(prisma: PrismaClient): Promise<TypeFacet[]> {
  const rows = await prisma.$queryRaw<{ slug: string | null; total: bigint }[]>`
    SELECT service_type AS slug, count(*) AS total
    FROM sermons WHERE service_type IS NOT NULL GROUP BY 1 ORDER BY 2 DESC`;
  return rows.map((r) => ({ slug: r.slug ?? "outro", total: num(r.total) }));
}

async function topicFacets(prisma: PrismaClient): Promise<TopicFacet[]> {
  const rows = await prisma.$queryRaw<
    { slug: string; nome: string; grupo: string; grupoNome: string; total: bigint }[]
  >`
    SELECT t.slug, t.name AS nome, t.group_slug AS grupo, t.group_name AS "grupoNome",
           count(st.sermon_id) AS total
    FROM topics t
    LEFT JOIN sermon_topics st ON st.topic_slug = t.slug
    GROUP BY t.slug, t.name, t.group_slug, t.group_name
    ORDER BY t.group_name, t.name`;

  return rows.map((r) => ({
    slug: r.slug,
    nome: r.nome,
    grupoSlug: r.grupo,
    grupoNome: r.grupoNome,
    total: num(r.total),
  }));
}

export async function facetTree(prisma: PrismaClient): Promise<FacetTree> {
  const [livros, series, pregadores, datas, tipos, temas] = await Promise.all([
    bookFacets(prisma),
    seriesFacets(prisma),
    preacherFacets(prisma),
    dateFacets(prisma),
    typeFacets(prisma),
    topicFacets(prisma),
  ]);
  return { livros, series, pregadores, datas, tipos, temas };
}
