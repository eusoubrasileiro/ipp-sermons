import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { loadSermons } from "../lib/corpus.ts";
import { parseCsv } from "../lib/csv.ts";
import { DATA_DIR } from "../lib/data-dir.ts";
import { loadBibleBooks } from "../lib/facets/bible.ts";
import { slugify } from "../lib/facets/slugify.ts";
import { type Problem, verdict } from "../lib/facets/verdict.ts";

/**
 * Pre-flight check on the committed facet files.
 *
 *   pnpm --filter @ipp/backend verify:facets
 *
 * Runs against the CSVs, not the database, so it works in CI with no Postgres
 * and no credential. It exists because every failure mode here is silent: a
 * series slug that matches nothing, a chapter beyond the end of its book, a
 * sermon id that no longer exists -- none of them raise an error, they just
 * render a page with nothing on it.
 *
 * The one check that earns its keep on every corpus update is the last: a NEW
 * series name that canonicalisation has never seen. That is the only thing a
 * human needs to look at after adding sermons, which is what keeps review to
 * two or three lines a quarter rather than a fifty-row sitting. It gets its own
 * exit code -- 0 clean, 1 blocked, 2 loadable but worth a look -- so a caller
 * can branch on it without grepping Portuguese out of stdout.
 */

const read = async (name: string) => parseCsv(await readFile(join(DATA_DIR, name), "utf8"));

async function main(): Promise<void> {
  const problems: Problem[] = [];
  const fail = (message: string) => problems.push({ severity: "erro", message });
  const warn = (message: string) => problems.push({ severity: "aviso", message });

  const { sermons } = loadSermons(await readFile(join(DATA_DIR, "metadata.csv"), "utf8"));
  const sermonIds = new Set(sermons.map((s) => s.id));

  const books = loadBibleBooks(await readFile(join(DATA_DIR, "facets/bible_books.csv"), "utf8"));
  const bookBySlug = new Map(books.map((b) => [b.slug, b]));

  const facets = await read("facets/sermon_facets.csv");
  const scriptures = await read("facets/sermon_scriptures.csv");
  const series = await read("facets/series.csv");

  // --- reference data -------------------------------------------------------
  if (books.length !== 66) fail(`bible_books.csv tem ${books.length} livros, esperado 66`);

  // --- series ---------------------------------------------------------------
  const seriesSlugs = new Set(series.map((r) => (r.slug ?? "").trim()));
  const variantToSlug = new Map<string, string>();
  for (const r of series) {
    const slug = (r.slug ?? "").trim();
    variantToSlug.set(slug, slug);
    for (const v of (r.variants ?? "").split("|")) {
      const name = v.trim();
      if (name) variantToSlug.set(slugify(name), slug);
    }
  }

  for (const r of series) {
    const parent = (r.parent_slug ?? "").trim();
    if (parent && !seriesSlugs.has(parent)) {
      fail(`série "${r.name}" aponta para série-mãe inexistente: ${parent}`);
    }
    if ((r.slug ?? "").trim() && parent === (r.slug ?? "").trim()) {
      fail(`série "${r.name}" é mãe de si mesma`);
    }
  }

  // --- sermon facets --------------------------------------------------------
  const seenSermons = new Set<string>();
  const unknownSeries = new Map<string, string>();

  for (const r of facets) {
    const id = (r.sermon_id ?? "").trim();
    if (!sermonIds.has(id)) fail(`sermon_facets.csv referencia sermão inexistente: ${id}`);
    if (seenSermons.has(id)) fail(`sermon_facets.csv tem ${id} duas vezes`);
    seenSermons.add(id);

    const raw = (r.series_slug ?? "").trim();
    if (raw && !variantToSlug.has(raw)) {
      unknownSeries.set(raw, (r.series_candidate ?? raw).trim());
    }
  }

  for (const s of sermons) {
    if (!seenSermons.has(s.id)) fail(`sermão sem facetas derivadas: ${s.title}`);
  }

  // --- scripture ------------------------------------------------------------
  const scriptureKeys = new Set<string>();
  for (const r of scriptures) {
    const id = (r.sermon_id ?? "").trim();
    const slug = (r.book_slug ?? "").trim();
    const chapter = Number.parseInt((r.chapter ?? "0").trim(), 10) || 0;

    if (!sermonIds.has(id)) fail(`sermon_scriptures.csv referencia sermão inexistente: ${id}`);

    const book = bookBySlug.get(slug);
    if (!book) {
      fail(`sermon_scriptures.csv referencia livro desconhecido: ${slug}`);
      continue;
    }
    if (chapter < 0 || chapter > book.chapters) {
      fail(`${slug} ${chapter} está fora do livro (${book.chapters} capítulos)`);
    }

    const key = `${id}|${slug}|${chapter}`;
    if (scriptureKeys.has(key)) fail(`sermon_scriptures.csv duplica ${key}`);
    scriptureKeys.add(key);
  }

  // --- the check that matters on every corpus update ------------------------
  if (unknownSeries.size > 0) {
    for (const [slug, name] of unknownSeries) {
      warn(`série nova, ainda não canonizada: "${name}" (${slug})`);
    }
    warn("rode `pnpm --filter @ipp/backend canonicalize:series` para incorporá-las");
  }

  // --- report ---------------------------------------------------------------
  const withScripture = new Set(scriptures.map((r) => (r.sermon_id ?? "").trim())).size;
  const pct = ((100 * withScripture) / sermons.length).toFixed(1);

  console.log(`sermões            ${sermons.length}`);
  console.log(`com passagem       ${withScripture} (${pct}%)`);
  console.log(`linhas de escritura ${scriptures.length}`);
  console.log(`séries             ${series.length}`);
  console.log(`livros             ${books.length}`);

  const { errors, warnings, exitCode } = verdict(problems);

  for (const p of warnings) console.log(`AVISO  ${p.message}`);
  for (const p of errors) console.error(`ERRO   ${p.message}`);

  if (errors.length > 0) console.error(`\n${errors.length} problema(s) bloqueante(s).`);
  else console.log(`\nOK${warnings.length ? ` (${warnings.length} aviso(s))` : ""}`);

  // 0 clean, 1 blocked, 2 loadable but a human should look — see verdict.ts.
  process.exit(exitCode);
}

await main();
