import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { loadSermons, readTranscript } from "../lib/corpus.ts";
import { parseCsv } from "../lib/csv.ts";
import { DATA_DIR } from "../lib/data-dir.ts";
import { cliLimit, reportFailures, runBatch } from "../lib/facets/batch.ts";
import { loadBibleBooks } from "../lib/facets/bible.ts";
import { type CsvValue, writeCsv } from "../lib/facets/csv.ts";
import {
  buildPrompt,
  buildSchema,
  decisionToRows,
  EXTRACT_SYSTEM,
  type ScriptureDecision,
} from "../lib/facets/extract-prompt.ts";
import { createOpenRouterLlm, LLM_MODEL } from "../lib/llm.ts";
import { createUsageMeter } from "../lib/usage.ts";

/**
 * Finds the passage for the sermons whose title never named one.
 *
 *   pnpm --filter @ipp/backend extract:scripture [--limit N] [--dry-run]
 *
 * The title parser covers 73% of the corpus for free. The rest opens with the
 * preacher reading the text aloud, so this reads the first 2500 words and asks
 * a model — and, just as importantly, lets it answer "none", which is the
 * correct answer for the catechism and conference material in the tail.
 *
 * Every decision is cached in `data/facets/scripture_llm.csv` and committed.
 * That file, not this script, is the ground truth: re-running costs nothing for
 * sermons already decided, an interrupted run resumes where it stopped, and a
 * wrong call can be corrected by editing one line instead of re-spending.
 *
 * `sermon_scriptures.csv` is then rebuilt as (title rows) + (cached decisions),
 * so it must be regenerated after `derive:facets`, which writes title rows only.
 */

/** What this run spends, reported by OpenRouter rather than estimated. */
const meter = createUsageMeter();
const DRY_RUN = process.argv.includes("--dry-run");
/** Flushed this often so a crash costs a handful of calls, not the whole run. */
const FLUSH_EVERY = 20;

const SCRIPTURE_COLUMNS = [
  "sermon_id",
  "book_slug",
  "chapter",
  "verse_start",
  "verse_end",
  "source",
  "is_primary",
];

const CACHE_COLUMNS = [
  "sermon_id",
  "livro",
  "capitulo_inicio",
  "versiculo_inicio",
  "capitulo_fim",
  "versiculo_fim",
  "justificativa",
];

const LIMIT = cliLimit();

const num = (v: string | undefined): number | null => {
  const parsed = Number.parseInt(v ?? "", 10);
  return Number.isFinite(parsed) ? parsed : null;
};

async function readCache(path: string): Promise<Map<string, ScriptureDecision>> {
  const text = await readFile(path, "utf8").catch(() => "");
  const cache = new Map<string, ScriptureDecision>();

  for (const row of text ? parseCsv(text) : []) {
    if (!row.sermon_id) continue;
    cache.set(row.sermon_id, {
      livro: row.livro || null,
      capitulo_inicio: num(row.capitulo_inicio),
      versiculo_inicio: num(row.versiculo_inicio),
      capitulo_fim: num(row.capitulo_fim),
      versiculo_fim: num(row.versiculo_fim),
      justificativa: row.justificativa ?? "",
    });
  }

  return cache;
}

const cacheRows = (cache: Map<string, ScriptureDecision>): Record<string, CsvValue>[] =>
  [...cache].map(([sermon_id, d]) => ({ sermon_id, ...d }));

async function main(): Promise<void> {
  const books = loadBibleBooks(await readFile(join(DATA_DIR, "facets/bible_books.csv"), "utf8"));
  const { sermons } = loadSermons(await readFile(join(DATA_DIR, "metadata.csv"), "utf8"));

  const scripturePath = join(DATA_DIR, "facets/sermon_scriptures.csv");
  const cachePath = join(DATA_DIR, "facets/scripture_llm.csv");

  const existing = parseCsv(await readFile(scripturePath, "utf8"));
  const titleRows = existing.filter((r) => r.source === "titulo");
  const fromTitle = new Set(titleRows.map((r) => r.sermon_id));

  const cache = await readCache(cachePath);
  const todo = sermons.filter((s) => !fromTitle.has(s.id) && !cache.has(s.id)).slice(0, LIMIT);

  console.log(`sermons              ${sermons.length}`);
  console.log(`passage from title   ${fromTitle.size}`);
  console.log(`already decided      ${cache.size}`);
  console.log(`to ask the model     ${todo.length}  (${LLM_MODEL})`);

  if (DRY_RUN) {
    for (const s of todo.slice(0, 20)) console.log(`  ${s.title}`);
    console.log("\n--dry-run: stopping before the LLM calls.");
    return;
  }

  if (todo.length > 0) {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) throw new Error("OPENROUTER_API_KEY is not set");
    await askModel(books, todo, cache, apiKey, cachePath);
  }

  await writeFile(cachePath, writeCsv(CACHE_COLUMNS, cacheRows(cache)));
  await writeScriptures(books, titleRows, cache, scripturePath, sermons.length);
}

async function askModel(
  books: ReturnType<typeof loadBibleBooks>,
  todo: { id: string; title: string; transcriptFile: string }[],
  cache: Map<string, ScriptureDecision>,
  apiKey: string,
  cachePath: string,
): Promise<void> {
  const llm = createOpenRouterLlm({ apiKey, onUsage: meter.record });
  const schema = buildSchema(books);
  const failures = await runBatch(
    todo,
    async (sermon) => {
      const transcript = await readTranscript(DATA_DIR, sermon.transcriptFile);
      const decision = await llm.complete<ScriptureDecision>({
        system: EXTRACT_SYSTEM,
        user: buildPrompt(sermon.title, transcript),
        schema,
        schemaName: "passagem_biblica",
      });
      cache.set(sermon.id, decision);
    },
    {
      label: (sermon) => sermon.title,
      flushEvery: FLUSH_EVERY,
      onFlush: async (done, total) => {
        await writeFile(cachePath, writeCsv(CACHE_COLUMNS, cacheRows(cache)));
        console.log(`  ${done}/${total}…`);
      },
    },
  );

  reportFailures(failures);
}

async function writeScriptures(
  books: ReturnType<typeof loadBibleBooks>,
  titleRows: Record<string, string>[],
  cache: Map<string, ScriptureDecision>,
  path: string,
  total: number,
): Promise<void> {
  const llmRows = [...cache].flatMap(([id, d]) => decisionToRows(books, id, d));
  const rows = [...titleRows, ...llmRows];

  await writeFile(path, writeCsv(SCRIPTURE_COLUMNS, rows));

  const covered = new Set(rows.map((r) => String(r.sermon_id))).size;
  const noPassage = [...cache.values()].filter((d) => !d.livro).length;

  console.log(`\nwrote data/facets/scripture_llm.csv and data/facets/sermon_scriptures.csv`);
  console.log(`  decided as "no passage"  ${noPassage}`);
  console.log(`  rows from the transcript ${llmRows.length}`);
  console.log(
    `  coverage                 ${covered}/${total} (${((100 * covered) / total).toFixed(1)}%)`,
  );
}

await main();

// Reported by OpenRouter, not multiplied out of a price list: the list goes
// stale without anyone noticing. Silent when the run made no paid call.
const spend = meter.summary();
if (spend) console.log(`\n${spend}`);
