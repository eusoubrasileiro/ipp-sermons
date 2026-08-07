import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { loadSermons, parseCsv } from "../lib/corpus.ts";
import { clusterNames, type NameCluster } from "../lib/facets/cluster.ts";
import { writeCsv } from "../lib/facets/csv.ts";
import {
  buildSeriesRows,
  retiredSlugs,
  SERIES_COLUMNS,
  type SeriesDecision,
} from "../lib/facets/series-taxonomy.ts";
import { createOpenRouterLlm, LLM_MODEL_STRONG } from "../lib/llm.ts";

/**
 * Turns the raw series names the title parser produced into a canonical,
 * committed taxonomy: `data/facets/series.csv`.
 *
 *   pnpm --filter @ipp/backend canonicalize:series [--dry-run]
 *
 * Three stages, cheapest first:
 *
 *   1. Fuzzy clustering (`lib/facets/cluster.ts`) folds spelling drift together
 *      -- "Atribututos de Deus" beside "Atributos de Deus" -- for free.
 *   2. One LLM call over the whole clustered list picks the correct spelling,
 *      names each Westminster chapter, and merges anything stage 1 missed.
 *      Fifty names fit comfortably in one prompt, so this costs cents.
 *   3. The result is written as CSV and committed. Production never re-derives.
 *
 * `--dry-run` stops after stage 1 and prints the clusters, so the input to the
 * model can be inspected before spending anything.
 *
 * Safe to run unattended, because of stage 4: nothing is written if the answer
 * would retire a slug that is already committed. Growing the taxonomy is
 * harmless; merging away a live `/series/<slug>` is not, and the model has no
 * way to know which names somebody has linked to.
 */
const DATA_DIR = process.env.CORPUS_DIR ?? join(import.meta.dirname, "../../../data");
const DRY_RUN = process.argv.includes("--dry-run");

const SCHEMA = {
  type: "object",
  properties: {
    series: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "integer" },
          name: { type: "string" },
          description: { type: "string" },
          parent: { type: ["string", "null"] },
          merge_into: { type: ["integer", "null"] },
        },
        required: ["id", "name", "description", "parent", "merge_into"],
        additionalProperties: false,
      },
    },
  },
  required: ["series"],
  additionalProperties: false,
} as const;

const SYSTEM = `Você organiza o catálogo de séries de sermões de uma igreja presbiteriana brasileira (Igreja Presbiteriana Peregrinos).

Receberá uma lista de séries candidatas extraídas automaticamente dos títulos dos sermões. Cada candidata traz as grafias encontradas, quantos sermões tem e exemplos de títulos.

Para cada candidata devolva:
- name: o nome correto e legível da série, em português, com acentuação e capitalização corretas. Corrija erros de digitação evidentes (ex.: "Atribututos" -> "Atributos"). Para capítulos da Confissão de Fé de Westminster use o formato "CFW <n> — <nome do capítulo>", deduzindo o nome do capítulo dos títulos de exemplo.
- description: uma frase curta (máx. 120 caracteres) dizendo do que trata a série. Sem inventar fatos.
- parent: o nome da série-mãe quando a candidata é parte de um curso maior. Use exatamente "Confissão de Fé de Westminster" para todo CFW. Caso contrário null.
- merge_into: o id de outra candidata quando as duas são, de fato, o mesmo curso com nomes diferentes. Caso contrário null.

Regras rígidas:
- NUNCA junte capítulos diferentes da CFW (CFW 3 e CFW 8 são cursos distintos).
- Só use merge_into quando tiver certeza de que é o mesmo curso.
- Não invente séries que não estão na lista.
- Devolva exatamente uma entrada para cada id recebido.`;

async function main(): Promise<void> {
  const csvText = await readFile(join(DATA_DIR, "metadata.csv"), "utf8");
  const { sermons } = loadSermons(csvText);
  const titleById = new Map(sermons.map((s) => [s.id, s.title]));

  const facets = parseCsv(await readFile(join(DATA_DIR, "facets/sermon_facets.csv"), "utf8"));

  const counts = new Map<string, number>();
  const examples = new Map<string, string[]>();
  for (const row of facets) {
    const name = (row.series_candidate ?? "").trim();
    if (!name) continue;
    counts.set(name, (counts.get(name) ?? 0) + 1);
    const seen = examples.get(name) ?? [];
    const title = titleById.get(row.sermon_id ?? "");
    if (title && seen.length < 3) seen.push(title);
    examples.set(name, seen);
  }

  const clusters = clusterNames([...counts].map(([name, count]) => ({ name, count })));
  console.log(`${counts.size} raw names -> ${clusters.length} clusters after fuzzy merge`);

  const merged = clusters.filter((c) => c.members.length > 1);
  for (const c of merged) {
    console.log(`  merged: ${c.members.join("  |  ")}`);
  }

  if (DRY_RUN) {
    console.log("\n--dry-run: stopping before the LLM call.");
    for (const c of clusters) console.log(`  ${String(c.count).padStart(3)}  ${c.provisional}`);
    return;
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is not set");

  const prompt = clusters
    .map((c, id) => {
      const sample = c.members.flatMap((m) => examples.get(m) ?? []).slice(0, 3);
      return [
        `id: ${id}`,
        `grafias: ${c.members.join(" | ")}`,
        `sermões: ${c.count}`,
        `exemplos: ${sample.join(" ;; ")}`,
      ].join("\n");
    })
    .join("\n\n");

  const llm = createOpenRouterLlm({ apiKey, model: LLM_MODEL_STRONG });
  const answer = await llm.complete<{ series: SeriesDecision[] }>({
    system: SYSTEM,
    user: prompt,
    schema: SCHEMA,
    schemaName: "series_taxonomy",
  });

  await writeSeries(clusters, answer.series, DATA_DIR);
}

/**
 * The slugs in the last commit — the ones whose URLs are live.
 *
 * Read from git rather than from disk: the file on disk is what this run is
 * about to overwrite, and on a second run in the same session it would already
 * carry the model's previous answer, which protects nothing.
 */
function committedSlugs(dataDir: string): string[] {
  try {
    const csv = execFileSync("git", ["show", "HEAD:data/facets/series.csv"], {
      cwd: dataDir,
      encoding: "utf8",
    });
    return parseCsv(csv)
      .map((r) => (r.slug ?? "").trim())
      .filter(Boolean);
  } catch {
    return []; // Not committed yet — there is nothing live to protect.
  }
}

async function writeSeries(
  clusters: NameCluster[],
  decisions: SeriesDecision[],
  dataDir: string,
): Promise<void> {
  const rows = buildSeriesRows(clusters, decisions);

  const retired = retiredSlugs(
    committedSlugs(dataDir),
    rows.map((r) => r.slug),
  );
  if (retired.length > 0) {
    console.error(`\n${retired.length} committed series slug(s) would be retired:`);
    for (const slug of retired) console.error(`  /series/${slug}`);
    console.error("\nNothing was written. Those URLs are live; re-run with the merge");
    console.error("reviewed by hand, or accept the loss by editing series.csv yourself.");
    process.exit(1);
  }

  await writeFile(join(dataDir, "facets/series.csv"), writeCsv(SERIES_COLUMNS, rows));

  const real = rows.filter((r) => r.sermon_count >= 2);
  console.log("\nwrote data/facets/series.csv");
  console.log(`  ${rows.length} series, ${real.length} with 2+ sermons`);
  for (const r of real) {
    const parent = r.parent_name ? `  (${r.parent_name})` : "";
    console.log(`  ${String(r.sermon_count).padStart(3)}  ${r.name}${parent}`);
  }
}

await main();
