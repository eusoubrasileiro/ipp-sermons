import { readFile } from "node:fs/promises";
import { join } from "node:path";
import pkg from "@prisma/client";
import { parseCsv } from "../lib/csv.ts";
import { DATA_DIR } from "../lib/data-dir.ts";
import { assertMatched, int } from "../lib/facets/csv.ts";
import { scripturePayload, spotifyPartition, topicPayload } from "../lib/facets/load-payload.ts";
import { buildVariantIndex, resolveSeries } from "../lib/facets/variants.ts";

/**
 * Loads the committed facet tables from data/facets/ into Postgres.
 *
 *   pnpm --filter @ipp/backend index:facets
 *
 * Separate from `index-corpus` on purpose. Indexing the corpus costs money and
 * takes minutes because it embeds every chunk; loading facets is free, takes a
 * second, and is the thing that changes whenever a title parser or an LLM pass
 * is re-run. Tying them together would mean paying to re-embed 20,000 chunks
 * to fix one series name.
 *
 * Idempotent: reference tables are upserts keyed on the natural key, and the
 * two derived tables are replaced wholesale inside a transaction, so a partial
 * run cannot empty the site.
 */
const { PrismaClient } = pkg;

const facetsFile = (name: string) => join(DATA_DIR, "facets", name);
const readFacets = async (name: string) => parseCsv(await readFile(facetsFile(name), "utf8"));

const prisma = new PrismaClient();

async function loadBooks(): Promise<number> {
  const rows = await readFacets("bible_books.csv");
  for (const r of rows) {
    const data = {
      name: (r.name ?? "").trim(),
      testament: (r.testament ?? "").trim(),
      canonOrder: int(r.order) ?? 0,
      chapters: int(r.chapters) ?? 0,
    };
    await prisma.bibleBook.upsert({
      where: { slug: (r.slug ?? "").trim() },
      create: { slug: (r.slug ?? "").trim(), ...data },
      update: data,
    });
  }
  return rows.length;
}

/**
 * Two passes over the same file.
 *
 * `parent_slug` is a self-reference, so every row has to exist before any of
 * them can point at another. Inserting parents first would work only if the
 * file happened to be ordered that way, and it is ordered by sermon count.
 */
async function loadSeries(): Promise<number> {
  const rows = await readFacets("series.csv");

  for (const r of rows) {
    const slug = (r.slug ?? "").trim();
    const data = {
      name: (r.name ?? "").trim(),
      kind: (r.kind ?? "").trim(),
      description: (r.description ?? "").trim(),
      sermonCount: int(r.sermon_count) ?? 0,
    };
    await prisma.series.upsert({ where: { slug }, create: { slug, ...data }, update: data });
  }

  for (const r of rows) {
    const parentSlug = (r.parent_slug ?? "").trim();
    if (!parentSlug) continue;
    const known = await prisma.series.findUnique({ where: { slug: parentSlug } });
    if (!known) continue; // A grouping label with no row of its own.
    await prisma.series.update({
      where: { slug: (r.slug ?? "").trim() },
      data: { parentSlug },
    });
  }

  return rows.length;
}

async function loadTopics(): Promise<number> {
  let rows: Record<string, string>[] = [];
  try {
    rows = await readFacets("taxonomy.csv");
  } catch {
    return 0; // The topic pass has not been run yet.
  }

  for (const r of rows) {
    const slug = (r.topico_slug ?? "").trim();
    if (!slug) continue;
    const data = {
      name: (r.topico_nome ?? "").trim(),
      groupSlug: (r.grupo_slug ?? "").trim(),
      groupName: (r.grupo_nome ?? "").trim(),
      description: (r.descricao ?? "").trim(),
    };
    await prisma.topic.upsert({ where: { slug }, create: { slug, ...data }, update: data });
  }
  return rows.length;
}

async function loadSermonFacets(): Promise<{ updated: number; missing: string[] }> {
  const rows = await readFacets("sermon_facets.csv");
  const known = new Set(
    (await prisma.series.findMany({ select: { slug: true } })).map((s) => s.slug),
  );
  const variants = buildVariantIndex(await readFacets("series.csv"));

  let updated = 0;
  const missing: string[] = [];

  for (const r of rows) {
    const id = (r.sermon_id ?? "").trim();
    const seriesSlug = (r.series_slug ?? "").trim();
    const canonical = resolveSeries(variants, seriesSlug);
    // A sermon whose series was dropped in canonicalisation keeps its other
    // facets rather than failing the whole load on a foreign key.
    const resolved = canonical && known.has(canonical) ? canonical : null;
    if (seriesSlug && !resolved) missing.push(seriesSlug);

    const result = await prisma.sermon.updateMany({
      where: { id },
      data: {
        serviceType: (r.service_type ?? "").trim() || null,
        seriesSlug: resolved,
        seriesPart: int(r.series_part),
        displayTitle: (r.display_title ?? "").trim() || null,
      },
    });
    updated += result.count;
  }

  return { updated, missing: [...new Set(missing)] };
}

/** Sermons absent from the file keep the column's `true` default: no episode, so the flag is moot. */
async function loadSpotifyLiveness(): Promise<{ alive: number; dead: number }> {
  const ids = spotifyPartition(await readFacets("spotify_episodes.csv"));
  const set = async (list: string[], spotifyAlive: boolean) =>
    (await prisma.sermon.updateMany({ where: { id: { in: list } }, data: { spotifyAlive } })).count;

  const counts = { alive: await set(ids.alive, true), dead: await set(ids.dead, false) };
  assertMatched("a sermon id", ids.alive.length + ids.dead.length, counts.alive + counts.dead);
  return counts;
}

async function loadScriptures(): Promise<{ rows: number; skipped: number }> {
  const rows = await readFacets("sermon_scriptures.csv");
  const sermons = new Set(
    (await prisma.sermon.findMany({ select: { id: true } })).map((s) => s.id),
  );
  const books = new Set(
    (await prisma.bibleBook.findMany({ select: { slug: true } })).map((b) => b.slug),
  );

  // Replace wholesale: the derivation is authoritative, and a chapter that
  // stopped being derived must stop being listed. Build and validate the
  // payload before the delete, then do both in one transaction — the old order
  // emptied the table and only then threw, on a production deploy.
  const payload = scripturePayload(rows, sermons, books);

  await prisma.$transaction([
    prisma.sermonScripture.deleteMany({}),
    prisma.sermonScripture.createMany({ data: payload, skipDuplicates: true }),
  ]);
  return { rows: payload.length, skipped: rows.length - payload.length };
}

async function loadSermonTopics(): Promise<number> {
  let rows: Record<string, string>[] = [];
  try {
    rows = await readFacets("sermon_topics.csv");
  } catch {
    return 0;
  }

  const topics = new Set(
    (await prisma.topic.findMany({ select: { slug: true } })).map((t) => t.slug),
  );
  const sermons = new Set(
    (await prisma.sermon.findMany({ select: { id: true } })).map((s) => s.id),
  );

  const payload = topicPayload(rows, sermons, topics);

  await prisma.$transaction([
    prisma.sermonTopic.deleteMany({}),
    prisma.sermonTopic.createMany({ data: payload, skipDuplicates: true }),
  ]);
  return payload.length;
}

async function main(): Promise<void> {
  console.log(`loading facets from ${DATA_DIR}/facets`);

  const books = await loadBooks();
  const series = await loadSeries();
  const topics = await loadTopics();
  const { updated, missing } = await loadSermonFacets();
  const scriptures = await loadScriptures();
  const sermonTopics = await loadSermonTopics();
  const spotify = await loadSpotifyLiveness();

  console.log(`  bible_books        ${books}`);
  console.log(`  series             ${series}`);
  console.log(
    `  topics             ${topics}${topics === 0 ? "  (taxonomy.csv not present yet)" : ""}`,
  );
  console.log(`  sermons updated    ${updated}`);
  console.log(
    `  sermon_scriptures  ${scriptures.rows}${scriptures.skipped ? `  (${scriptures.skipped} skipped: unknown sermon or book)` : ""}`,
  );
  console.log(`  sermon_topics      ${sermonTopics}`);
  console.log(`  spotify episodes   ${spotify.alive} alive, ${spotify.dead} aged out of the feed`);

  if (missing.length > 0) {
    console.log(
      `\n  ${missing.length} series slug(s) in sermon_facets.csv have no row in series.csv:`,
    );
    for (const slug of missing.slice(0, 10)) console.log(`    ${slug}`);
    console.log("  Re-run canonicalize:series, or these sermons browse without a series.");
  }
}

try {
  await main();
} finally {
  await prisma.$disconnect();
}
