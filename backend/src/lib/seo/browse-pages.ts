import { type SearchFilters, SITE_TITLE } from "@ipp/shared";
import type { PrismaClient } from "@prisma/client";
import { type FacetTree, facetTree } from "../browse/facets.ts";
import { sermonWhere } from "../browse/list.ts";
import type { SeoPage } from "./html.ts";
import { type FacetLink, type ListedSermon, listingPage } from "./listing-page.ts";

/**
 * The browse tree, prerendered.
 *
 * Every facet page is currently the same empty shell, so the archive has one
 * indexable URL and 560 unreachable ones. These pages are the crawl path: real
 * `<a href>` from an index to a facet to a sermon, so the corpus is reachable
 * without JavaScript and without trusting the sitemap alone.
 *
 * Labels come from `facetTree()` rather than from new queries — it is the same
 * tree the SPA renders, so a heading here and a heading there cannot disagree.
 * That costs eight GROUP BYs per facet page, which is the right trade for five
 * index pages and a few hundred leaves that a crawler visits rarely.
 */

export const FAMILIES = ["biblia", "temas", "series", "pregadores", "datas"] as const;
type Family = (typeof FAMILIES)[number];

const FAMILY_TITLE: Record<Family, string> = {
  biblia: "Bíblia",
  temas: "Temas",
  series: "Séries",
  pregadores: "Pregadores",
  datas: "Datas",
};

const FAMILY_BLURB: Record<Family, string> = {
  biblia: "Sermões da Igreja Presbiteriana Peregrinos por livro e capítulo da Bíblia.",
  temas: "Sermões da Igreja Presbiteriana Peregrinos por tema.",
  series: "Séries e cursos pregados na Igreja Presbiteriana Peregrinos.",
  pregadores: "Todos os pregadores do acervo da Igreja Presbiteriana Peregrinos.",
  datas: "O acervo da Igreja Presbiteriana Peregrinos ano a ano.",
};

/**
 * How many sermons a leaf lists.
 *
 * Capped rather than paginated: the largest facet in this archive is one
 * preacher with a few hundred sermons, which is ~30 KB of links, and paginating
 * would invite a crawler to walk `?pagina=N` and pay a query for each.
 */
const LEAF_CAP = 500;

const MONTHS = [
  "janeiro",
  "fevereiro",
  "março",
  "abril",
  "maio",
  "junho",
  "julho",
  "agosto",
  "setembro",
  "outubro",
  "novembro",
  "dezembro",
];

type Leaf = { heading: string; filters: SearchFilters };

function bibleLeaf(tree: FacetTree, slug: string, sub: string | undefined): Leaf | null {
  const book = tree.livros.find((b) => b.slug === slug);
  if (!book) return null;
  if (sub === undefined) return { heading: book.nome, filters: { livros: [slug] } };

  const chapter = Number.parseInt(sub, 10);
  // 150 is the Psalms; SearchFiltersSchema uses the same ceiling.
  if (!Number.isInteger(chapter) || chapter < 1 || chapter > 150) return null;
  return { heading: `${book.nome} ${chapter}`, filters: { livros: [slug], capitulo: chapter } };
}

/** `/datas/2024` and `/datas/2024/3` become a date range, not a lookup. */
function dateLeaf(slug: string, sub: string | undefined): Leaf | null {
  const year = Number.parseInt(slug, 10);
  if (!/^\d{4}$/.test(slug) || year < 2000 || year > 2100) return null;
  if (sub === undefined) {
    return { heading: String(year), filters: { de: `${year}-01-01`, ate: `${year}-12-31` } };
  }

  const month = Number.parseInt(sub, 10);
  if (!Number.isInteger(month) || month < 1 || month > 12) return null;
  const pad = String(month).padStart(2, "0");
  // The last day is the day before the first of the next month, which sidesteps
  // both February and December rolling over.
  const end = new Date(Date.UTC(year, month, 0)).toISOString().slice(0, 10);
  return {
    heading: `${MONTHS[month - 1]} de ${year}`,
    filters: { de: `${year}-${pad}-01`, ate: end },
  };
}

function leafOf(
  tree: FacetTree,
  family: Family,
  slug: string,
  sub: string | undefined,
): Leaf | null {
  if (family === "biblia") return bibleLeaf(tree, slug, sub);
  if (family === "datas") return dateLeaf(slug, sub);
  if (sub !== undefined) return null;

  if (family === "temas") {
    const topic = tree.temas.find((t) => t.slug === slug);
    return topic ? { heading: topic.nome, filters: { temas: [slug] } } : null;
  }
  if (family === "series") {
    const series = tree.series.find((s) => s.slug === slug);
    return series ? { heading: series.nome, filters: { series: [slug] } } : null;
  }

  const preacher = tree.pregadores.find((p) => p.slug === slug);
  // `artist` is the raw column and the only value the filter matches on.
  return preacher ? { heading: preacher.artist, filters: { pregadores: [preacher.artist] } } : null;
}

function indexLinks(tree: FacetTree, family: Family): FacetLink[] {
  if (family === "biblia") {
    return tree.livros.map((b) => ({ href: `/biblia/${b.slug}`, label: b.nome, total: b.total }));
  }
  if (family === "temas") {
    return tree.temas
      .filter((t) => t.total > 0)
      .map((t) => ({ href: `/temas/${t.slug}`, label: t.nome, total: t.total }));
  }
  if (family === "series") {
    return tree.series
      .filter((s) => s.total > 0)
      .map((s) => ({ href: `/series/${s.slug}`, label: s.nome, total: s.total }));
  }
  if (family === "pregadores") {
    return tree.pregadores.map((p) => ({
      href: `/pregadores/${p.slug}`,
      label: p.artist,
      total: p.total,
    }));
  }
  return tree.datas.map((d) => ({ href: `/datas/${d.ano}`, label: String(d.ano), total: d.total }));
}

async function listedSermons(
  prisma: PrismaClient,
  filters: SearchFilters,
): Promise<{ total: number; sermons: ListedSermon[] }> {
  const where = sermonWhere(filters);
  const [total, rows] = await Promise.all([
    prisma.sermon.count({ where }),
    prisma.sermon.findMany({
      where,
      orderBy: [{ date: "desc" }],
      take: LEAF_CAP,
      select: { id: true, title: true, artist: true, date: true },
    }),
  ]);

  return {
    total,
    sermons: rows.map((row) => ({
      id: row.id,
      title: row.title,
      artist: row.artist,
      date: row.date.toISOString().slice(0, 10),
    })),
  };
}

export async function facetIndexPage(prisma: PrismaClient, family: Family): Promise<SeoPage> {
  const links = indexLinks(await facetTree(prisma), family);
  return listingPage({
    heading: FAMILY_TITLE[family],
    title: `${FAMILY_TITLE[family]} — Sermões IPP`,
    description: FAMILY_BLURB[family],
    path: `/${family}`,
    sermons: [],
    links,
  });
}

export async function facetLeafPage(
  prisma: PrismaClient,
  family: Family,
  slug: string,
  sub?: string | undefined,
): Promise<SeoPage | null> {
  const leaf = leafOf(await facetTree(prisma), family, slug, sub);
  if (!leaf) return null;

  const { total, sermons } = await listedSermons(prisma, leaf.filters);
  const path = sub === undefined ? `/${family}/${slug}` : `/${family}/${slug}/${sub}`;

  return listingPage({
    heading: leaf.heading,
    title: `${leaf.heading} — Sermões IPP`,
    // Distinct per facet on purpose: 560 pages sharing one sentence is what a
    // search engine collapses into a single result.
    description: `${total} ${total === 1 ? "sermão" : "sermões"} em ${leaf.heading}, do acervo da Igreja Presbiteriana Peregrinos.`,
    path,
    sermons,
    links: [],
    total,
  });
}

export async function homePage(prisma: PrismaClient): Promise<SeoPage> {
  const total = await prisma.sermon.count();

  return listingPage({
    heading: "Sermões da Igreja Presbiteriana Peregrinos",
    title: SITE_TITLE,
    description: `Busca em ${total} sermões transcritos da Igreja Presbiteriana Peregrinos: pesquise por tema, passagem bíblica ou pregador.`,
    path: "/",
    sermons: [],
    links: FAMILIES.map((family) => ({ href: `/${family}`, label: FAMILY_TITLE[family] })),
  });
}
