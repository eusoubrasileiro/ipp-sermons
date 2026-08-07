import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { loadSermons, parseCsv, readTranscript } from "../lib/corpus.ts";
import { DATA_DIR } from "../lib/data-dir.ts";
import { cliLimit, reportFailures, runBatch } from "../lib/facets/batch.ts";
import { type CsvValue, writeCsv } from "../lib/facets/csv.ts";
import {
  labelRows,
  MAX_TOPICS_PER_SERMON,
  SERMON_TOPIC_COLUMNS,
  sampleTranscript,
  type TopicLabel,
} from "../lib/facets/topics.ts";
import { createOpenRouterLlm, LLM_MODEL } from "../lib/llm.ts";
import { createUsageMeter } from "../lib/usage.ts";

/**
 * Labels every sermon against the committed taxonomy.
 *
 *   pnpm --filter @ipp/backend label:topics [--limit N] [--dry-run]
 *
 * The taxonomy is closed: the model picks slugs from an enum, so it cannot
 * invent a topic that would then break the foreign key at load time or build a
 * facet page that lists nothing.
 *
 * Resumable by output: a sermon already present in `sermon_topics.csv` is not
 * asked again, and the file is flushed as the run proceeds. A sermon the model
 * gives no topic at all would be re-asked on the next run -- rare enough with a
 * 56-topic taxonomy that a cache of negatives is not worth the extra file, and
 * the count is printed so the cost of a re-run is visible.
 */

/** What this run spends, reported by OpenRouter rather than estimated. */
const meter = createUsageMeter();
const DRY_RUN = process.argv.includes("--dry-run");
const FLUSH_EVERY = 40;

const LIMIT = cliLimit();

const SYSTEM = `Você classifica sermões de uma igreja presbiteriana brasileira (Igreja Presbiteriana Peregrinos) numa taxonomia FECHADA de temas.

Receberá o título, a passagem bíblica quando houver, e trechos da transcrição (início, meio e fim).

Devolva de 1 a ${MAX_TOPICS_PER_SERMON} tópicos, do mais central para o menos central, escolhidos EXCLUSIVAMENTE da lista fornecida.

- confianca: 1.0 quando o tópico é o assunto do sermão; 0.5 quando é tratado de forma relevante mas secundária; abaixo de 0.4 não devolva.
- Prefira poucos tópicos certos a muitos aproximados. Um sermão marcado com tudo que menciona não ajuda ninguém a encontrá-lo.
- Não escolha um tópico só porque a palavra aparece: o sermão precisa tratar do assunto.`;

type TaxonomyRow = {
  topico_slug: string;
  topico_nome: string;
  grupo_nome: string;
  descricao: string;
};

async function main(): Promise<void> {
  const taxonomy = parseCsv(
    await readFile(join(DATA_DIR, "facets/taxonomy.csv"), "utf8"),
  ) as unknown as TaxonomyRow[];

  if (taxonomy.length === 0) throw new Error("taxonomy.csv is empty — run propose:taxonomy first");
  const known = new Set(taxonomy.map((t) => t.topico_slug));

  const { sermons } = loadSermons(await readFile(join(DATA_DIR, "metadata.csv"), "utf8"));
  const outPath = join(DATA_DIR, "facets/sermon_topics.csv");

  const existingText = await readFile(outPath, "utf8").catch(() => "");
  const rows: Record<string, CsvValue>[] = existingText
    ? parseCsv(existingText).map((r) => ({
        sermon_id: r.sermon_id,
        topico_slug: r.topico_slug,
        confianca: Number(r.confianca ?? 0),
      }))
    : [];

  const done = new Set(rows.map((r) => String(r.sermon_id)));
  const todo = sermons.filter((s) => !done.has(s.id)).slice(0, LIMIT);

  const scriptures = await scriptureIndex();
  const catalogue = taxonomy
    .map((t) => `${t.topico_slug} — ${t.grupo_nome} > ${t.topico_nome}: ${t.descricao}`)
    .join("\n");

  console.log(`topics in taxonomy   ${known.size}`);
  console.log(`already labelled     ${done.size}`);
  console.log(`to label             ${todo.length}  (${LLM_MODEL})`);

  if (DRY_RUN) {
    console.log(`\n${catalogue}\n\n--dry-run: stopping before the LLM calls.`);
    return;
  }

  if (todo.length > 0) {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) throw new Error("OPENROUTER_API_KEY is not set");
    await label(todo, { known, catalogue, scriptures, apiKey, rows, outPath });
  }

  await writeFile(outPath, writeCsv(SERMON_TOPIC_COLUMNS, rows));
  report(rows, taxonomy, sermons.length);
}

async function scriptureIndex(): Promise<Map<string, string>> {
  const index = new Map<string, string>();
  const text = await readFile(join(DATA_DIR, "facets/sermon_scriptures.csv"), "utf8");

  for (const row of parseCsv(text)) {
    if (row.sermon_id && !index.has(row.sermon_id)) {
      index.set(row.sermon_id, `${row.book_slug} ${row.chapter === "0" ? "" : row.chapter}`.trim());
    }
  }

  return index;
}

async function label(
  todo: { id: string; title: string; transcriptFile: string }[],
  ctx: {
    known: Set<string>;
    catalogue: string;
    scriptures: Map<string, string>;
    apiKey: string;
    rows: Record<string, CsvValue>[];
    outPath: string;
  },
): Promise<void> {
  const llm = createOpenRouterLlm({ apiKey: ctx.apiKey, onUsage: meter.record });
  const schema = {
    type: "object",
    properties: {
      topicos: {
        type: "array",
        items: {
          type: "object",
          properties: {
            topico_slug: { type: "string", enum: [...ctx.known] },
            confianca: { type: "number" },
          },
          required: ["topico_slug", "confianca"],
          additionalProperties: false,
        },
      },
    },
    required: ["topicos"],
    additionalProperties: false,
  };

  const failures = await runBatch(
    todo,
    async (sermon) => {
      const transcript = await readTranscript(DATA_DIR, sermon.transcriptFile);
      const ref = ctx.scriptures.get(sermon.id);
      const answer = await llm.complete<{ topicos: TopicLabel[] }>({
        system: `${SYSTEM}\n\nTÓPICOS DISPONÍVEIS:\n${ctx.catalogue}`,
        user: `Título: ${sermon.title}${ref ? `\nPassagem: ${ref}` : ""}\n\nTranscrição (amostra):\n${sampleTranscript(transcript)}`,
        schema,
        schemaName: "temas_do_sermao",
      });
      ctx.rows.push(...labelRows(ctx.known, sermon.id, answer.topicos));
    },
    {
      label: (sermon) => sermon.title,
      flushEvery: FLUSH_EVERY,
      onFlush: async (done, total) => {
        await writeFile(ctx.outPath, writeCsv(SERMON_TOPIC_COLUMNS, ctx.rows));
        console.log(`  ${done}/${total}…`);
      },
    },
  );

  reportFailures(failures);
}

function report(rows: Record<string, CsvValue>[], taxonomy: TaxonomyRow[], total: number): void {
  const perTopic = new Map<string, number>();
  for (const r of rows) {
    const slug = String(r.topico_slug);
    perTopic.set(slug, (perTopic.get(slug) ?? 0) + 1);
  }

  const labelled = new Set(rows.map((r) => String(r.sermon_id))).size;
  const empty = taxonomy.filter((t) => !perTopic.has(t.topico_slug));

  console.log(`\nwrote data/facets/sermon_topics.csv`);
  console.log(`  sermons labelled   ${labelled}/${total}`);
  console.log(
    `  labels             ${rows.length}  (${(rows.length / labelled).toFixed(1)} per sermon)`,
  );
  console.log(`  topics never used  ${empty.length}`);

  const top = [...perTopic].sort((a, b) => b[1] - a[1]).slice(0, 12);
  const name = new Map(taxonomy.map((t) => [t.topico_slug, t.topico_nome]));
  for (const [slug, count] of top) {
    console.log(`    ${String(count).padStart(3)}  ${name.get(slug) ?? slug}`);
  }
  if (empty.length > 0) {
    console.log(`  vazios: ${empty.map((t) => t.topico_nome).join(" · ")}`);
  }
}

await main();

// Reported by OpenRouter, not multiplied out of a price list: the list goes
// stale without anyone noticing. Silent when the run made no paid call.
const spend = meter.summary();
if (spend) console.log(`\n${spend}`);
