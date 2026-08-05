import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { loadSermons, parseCsv } from "../lib/corpus.ts";
import { loadBibleBooks } from "../lib/facets/bible.ts";
import { type CsvValue, writeCsv } from "../lib/facets/csv.ts";
import { chaptersOf } from "../lib/facets/parse-scripture.ts";
import { parseTitle } from "../lib/facets/parse-title.ts";
import { slugify } from "../lib/facets/slugify.ts";

/**
 * Derives every facet a sermon title states outright, and writes it to
 * `data/facets/` for review and commit.
 *
 * Deliberately offline and deliberately deterministic: the corpus is ground
 * truth kept in git, so the facets derived from it are too. Production reads
 * the committed CSVs and never re-derives, which keeps a deploy from depending
 * on a parser change nobody reviewed.
 *
 *   pnpm --filter @ipp/backend derive:facets [--data-dir path]
 *
 * Writes:
 *   data/facets/sermon_facets.csv      one row per sermon
 *   data/facets/sermon_scriptures.csv  one row per (sermon, chapter)
 *
 * The series names here are *candidates*: raw, unmerged, still carrying the
 * corpus's typos. `canonicalize-series.ts` turns them into `series.csv`.
 */
const dataDirFlag = process.argv.indexOf("--data-dir");
const DATA_DIR =
  dataDirFlag !== -1 && process.argv[dataDirFlag + 1]
    ? (process.argv[dataDirFlag + 1] as string)
    : (process.env.CORPUS_DIR ?? join(import.meta.dirname, "../../../data"));

async function main(): Promise<void> {
  const books = loadBibleBooks(await readFile(join(DATA_DIR, "facets/bible_books.csv"), "utf8"));
  const csvText = await readFile(join(DATA_DIR, "metadata.csv"), "utf8");
  const { sermons } = loadSermons(csvText);

  // The SoundCloud description is the only place some conference names survive.
  const descriptions = new Map(parseCsv(csvText).map((r) => [r.txt, r.description ?? ""]));

  const facetRows: Record<string, CsvValue>[] = [];
  const scriptureRows: Record<string, CsvValue>[] = [];
  const seriesCounts = new Map<string, number>();

  for (const sermon of sermons) {
    const parsed = parseTitle(books, sermon.title, descriptions.get(sermon.transcriptFile));
    const seriesSlug = parsed.seriesCandidate ? slugify(parsed.seriesCandidate) : null;
    if (parsed.seriesCandidate) {
      seriesCounts.set(parsed.seriesCandidate, (seriesCounts.get(parsed.seriesCandidate) ?? 0) + 1);
    }

    facetRows.push({
      sermon_id: sermon.id,
      service_type: parsed.serviceType,
      series_candidate: parsed.seriesCandidate,
      series_slug: seriesSlug,
      series_part: parsed.part,
      event_name: parsed.eventName,
      display_title: parsed.displayTitle,
    });

    const ref = parsed.scripture;
    if (!ref) continue;

    const chapters = chaptersOf(ref);
    if (chapters.length === 0) {
      // The title names a book but no chapter -- still a usable facet, and
      // 0 rather than null so the row has a complete identity.
      scriptureRows.push({
        sermon_id: sermon.id,
        book_slug: ref.bookSlug,
        chapter: 0,
        verse_start: null,
        verse_end: null,
        source: "titulo",
        is_primary: true,
      });
      continue;
    }

    for (const chapter of chapters) {
      // Verses only bound the chapters they actually touch; a lesson on
      // "Gênesis 12-50" has no verse range for chapter 30.
      const isFirst = chapter === ref.chapterStart;
      const isLast = chapter === ref.chapterEnd;
      scriptureRows.push({
        sermon_id: sermon.id,
        book_slug: ref.bookSlug,
        chapter,
        verse_start: isFirst ? ref.verseStart : null,
        verse_end: isLast ? ref.verseEnd : null,
        source: "titulo",
        is_primary: true,
      });
    }
  }

  await writeFile(
    join(DATA_DIR, "facets/sermon_facets.csv"),
    writeCsv(
      [
        "sermon_id",
        "service_type",
        "series_candidate",
        "series_slug",
        "series_part",
        "event_name",
        "display_title",
      ],
      facetRows,
    ),
  );

  await writeFile(
    join(DATA_DIR, "facets/sermon_scriptures.csv"),
    writeCsv(
      ["sermon_id", "book_slug", "chapter", "verse_start", "verse_end", "source", "is_primary"],
      scriptureRows,
    ),
  );

  const byType = new Map<string, number>();
  for (const r of facetRows)
    byType.set(String(r.service_type), (byType.get(String(r.service_type)) ?? 0) + 1);
  const withBook = new Set(scriptureRows.map((r) => r.sermon_id)).size;
  const repeated = [...seriesCounts.values()].filter((c) => c >= 2).length;

  console.log(`sermons               ${sermons.length}`);
  console.log(`service types         ${[...byType].map(([t, c]) => `${t}=${c}`).join("  ")}`);
  console.log(
    `with a passage        ${withBook} (${((100 * withBook) / sermons.length).toFixed(1)}%)`,
  );
  console.log(`scripture rows        ${scriptureRows.length}`);
  console.log(`series candidates     ${seriesCounts.size} distinct, ${repeated} with 2+ parts`);
  console.log(`\nwrote data/facets/sermon_facets.csv and data/facets/sermon_scriptures.csv`);
}

await main();
