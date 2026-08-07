import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { loadSermons, parseCsv, readTranscript } from "../lib/corpus.ts";
import { DATA_DIR } from "../lib/data-dir.ts";
import { runBatch } from "../lib/facets/batch.ts";
import { agreementWith, divergent, type Labelled } from "../lib/facets/compare.ts";
import { type CsvValue, writeCsv } from "../lib/facets/csv.ts";
import {
  labelRows,
  sampleTranscript,
  TOPICS_SYSTEM,
  type TopicLabel,
  topicsPrompt,
  topicsSchema,
} from "../lib/facets/topics.ts";
import { createOpenRouterLlm, LLM_MODEL } from "../lib/llm.ts";
import { createUsageMeter } from "../lib/usage.ts";

/**
 * Labels a sample of already-labelled sermons several ways and reports where
 * the configurations disagree.
 *
 *   pnpm compare:topics --sample 40
 *
 * Two questions have to be answered before anything is reclassified: does
 * reading the whole transcript beat the three-window sample, and does a cheaper
 * model hold up in Portuguese. Neither is settled by argument.
 *
 * Nothing is written to the corpus. The output is a CSV of the rows that
 * differ, which is the short list a human would have to read -- typically a
 * third of the sample. Deciding which answer is better is `docs/facet-quality.md`
 * and needs a person; this only says where they differ, which needs nobody.
 *
 * Config `A` is the committed labelling, read from sermon_topics.csv rather
 * than re-run: it is what the corpus actually contains, and it is free.
 */
const OUT_DIR = join(import.meta.dirname, "../../../.compare");

type Config = {
  id: string;
  label: string;
  model?: string;
  /** How much of the transcript the model sees. */
  body: (transcript: string) => string;
};

/**
 * Only models advertising `structured_outputs` belong here. `response_format`
 * alone does not guarantee `strict: true`, and without that the closed enum
 * stops being a guarantee that the model cannot invent a topic.
 */
const CONFIGS: Config[] = [
  { id: "B", label: `${LLM_MODEL} + transcript inteiro`, body: (t) => t },
  {
    id: "C",
    label: "deepseek-v4-flash + transcript inteiro",
    model: "deepseek/deepseek-v4-flash-0731",
    body: (t) => t,
  },
  {
    id: "D",
    label: "ling-2.6-flash + transcript inteiro",
    model: "inclusionai/ling-2.6-flash",
    body: (t) => t,
  },
  { id: "E", label: `${LLM_MODEL} + amostra (como hoje)`, body: sampleTranscript },
];

const BASELINE = "A";
const ALL = [BASELINE, ...CONFIGS.map((c) => c.id)];

const meter = createUsageMeter();

function parseCliArgs(): { sample: number; dryRun: boolean } {
  const { values } = parseArgs({
    options: { sample: { type: "string" }, "dry-run": { type: "boolean", default: false } },
  });
  return {
    sample: Number.parseInt(values.sample ?? "40", 10),
    dryRun: values["dry-run"] ?? false,
  };
}

/**
 * An evenly spread sample, not the first N.
 *
 * metadata.csv is append-ordered, so the head of it is whatever was published
 * first. A stride samples across the whole corpus and stays the same between
 * runs, which is what makes two benches comparable.
 */
function spread<T>(items: T[], count: number): T[] {
  if (items.length <= count) return items;
  const stride = items.length / count;
  return Array.from({ length: count }, (_, i) => items[Math.floor(i * stride)] as T);
}

async function main(): Promise<void> {
  const args = parseCliArgs();

  const taxonomy = parseCsv(await readFile(join(DATA_DIR, "facets/taxonomy.csv"), "utf8"));
  const known = new Set(taxonomy.map((t) => (t.topico_slug ?? "").trim()).filter(Boolean));
  const catalogue = taxonomy
    .map((t) => `${t.topico_slug} — ${t.grupo_nome} > ${t.topico_nome}: ${t.descricao}`)
    .join("\n");

  const { sermons } = loadSermons(await readFile(join(DATA_DIR, "metadata.csv"), "utf8"));

  const committed = new Map<string, string[]>();
  for (const r of parseCsv(await readFile(join(DATA_DIR, "facets/sermon_topics.csv"), "utf8"))) {
    const id = (r.sermon_id ?? "").trim();
    committed.set(id, [...(committed.get(id) ?? []), (r.topico_slug ?? "").trim()]);
  }

  const scriptures = new Map<string, string>();
  for (const r of parseCsv(
    await readFile(join(DATA_DIR, "facets/sermon_scriptures.csv"), "utf8"),
  )) {
    const id = (r.sermon_id ?? "").trim();
    if (id && !scriptures.has(id)) {
      scriptures.set(id, `${r.book_slug} ${r.chapter === "0" ? "" : r.chapter}`.trim());
    }
  }

  // Only sermons the corpus already has an opinion about: without a baseline
  // there is nothing to compare a candidate against.
  const eligible = sermons.filter((s) => committed.has(s.id));
  const sample = spread(eligible, args.sample);

  console.log(`taxonomy            ${known.size} topics`);
  console.log(`labelled sermons    ${eligible.length}`);
  console.log(`sample              ${sample.length}`);
  for (const c of CONFIGS) console.log(`  ${c.id}  ${c.label}`);
  console.log(`  ${BASELINE}  committed labelling (free)`);

  if (args.dryRun) {
    console.log("\n--dry-run: stopping before the LLM calls.");
    return;
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is not set");

  const rows: Labelled[] = sample.map((s) => ({
    sermonId: s.id,
    title: s.title,
    byConfig: { [BASELINE]: committed.get(s.id) ?? [] },
  }));
  const byId = new Map(rows.map((r) => [r.sermonId, r]));

  for (const config of CONFIGS) {
    const llm = createOpenRouterLlm({
      apiKey,
      onUsage: meter.record,
      ...(config.model ? { model: config.model } : {}),
    });

    const failures = await runBatch(
      sample,
      async (sermon) => {
        const transcript = await readTranscript(DATA_DIR, sermon.transcriptFile);
        const answer = await llm.complete<{ topicos: TopicLabel[] }>({
          system: `${TOPICS_SYSTEM}\n\nTÓPICOS DISPONÍVEIS:\n${catalogue}`,
          user: topicsPrompt(sermon.title, scriptures.get(sermon.id), config.body(transcript)),
          schema: topicsSchema(known),
          schemaName: "temas_do_sermao",
        });
        const labelled = labelRows(known, sermon.id, answer.topicos);
        const row = byId.get(sermon.id);
        if (row) row.byConfig[config.id] = labelled.map((r) => String(r.topico_slug));
      },
      { label: (s) => s.title, concurrency: 4 },
    );

    console.log(`  ${config.id} done${failures.length ? `, ${failures.length} failed` : ""}`);
  }

  await report(rows, args.sample);
}

async function report(rows: Labelled[], sample: number): Promise<void> {
  const differ = divergent(rows, ALL);

  console.log(`\n${rows.length} sermons · ${ALL.length} configurations`);
  console.log(`  all agree           ${rows.length - differ.length}`);
  console.log(`  differ              ${differ.length}   <- the only ones worth reading`);

  console.log(`\nagreement with ${BASELINE} (what the corpus contains today):`);
  for (const a of agreementWith(rows, BASELINE, ALL)) {
    const pct = a.total === 0 ? 0 : Math.round((100 * a.exact) / a.total);
    console.log(
      `  ${a.config}  exact ${String(a.exact).padStart(3)}/${a.total} (${pct}%)   mean jaccard ${a.jaccard.toFixed(2)}`,
    );
  }

  await mkdir(OUT_DIR, { recursive: true });
  const stamp = new Date().toISOString().slice(0, 10);
  const path = join(OUT_DIR, `topics-${stamp}-n${sample}.csv`);

  const csv: Record<string, CsvValue>[] = differ.map((row) => ({
    sermon_id: row.sermonId,
    titulo: row.title,
    ...Object.fromEntries(ALL.map((c) => [`${c}_temas`, (row.byConfig[c] ?? []).join("|")])),
    melhor: "",
  }));

  await writeFile(
    path,
    writeCsv(["sermon_id", "titulo", ...ALL.map((c) => `${c}_temas`), "melhor"], csv),
  );

  console.log(`\nwrote ${path}`);
  console.log("Read the rows above and put A/B/C/D/E in `melhor` — see docs/facet-quality.md.");

  const spend = meter.summary();
  if (spend) console.log(`\n${spend}`);
}

await main();
