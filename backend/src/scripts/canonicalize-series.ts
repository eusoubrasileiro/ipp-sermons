import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { loadSermons, parseCsv } from "../lib/corpus.ts";
import { clusterNames, type NameCluster } from "../lib/facets/cluster.ts";
import { type CsvValue, writeCsv } from "../lib/facets/csv.ts";
import { slugify } from "../lib/facets/slugify.ts";
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
 * Re-running is safe: the output is a pure function of the corpus plus the
 * model's answer, and any hand-edit to series.csv survives because
 * `verify-facets` -- not this script -- is what runs in CI.
 */
const DATA_DIR = process.env.CORPUS_DIR ?? join(import.meta.dirname, "../../../data");
const DRY_RUN = process.argv.includes("--dry-run");

/** Deterministic; the model is not asked to guess what the title already says. */
function kindOf(name: string): string {
  if (/^CFW\s*\d/i.test(name)) return "cfw";
  if (/confer[êe]ncia/i.test(name)) return "conferencia";
  if (/congresso/i.test(name)) return "congresso";
  if (/confraria/i.test(name)) return "confraria";
  if (/^diaconia$/i.test(name)) return "diaconia";
  return "ebd";
}

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

type Adjudicated = {
  id: number;
  name: string;
  description: string;
  parent: string | null;
  merge_into: number | null;
};

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
  const answer = await llm.complete<{ series: Adjudicated[] }>({
    system: SYSTEM,
    user: prompt,
    schema: SCHEMA,
    schemaName: "series_taxonomy",
  });

  await writeSeries(clusters, answer.series, DATA_DIR);
}

async function writeSeries(
  clusters: NameCluster[],
  adjudicated: Adjudicated[],
  dataDir: string,
): Promise<void> {
  const byId = new Map(adjudicated.map((a) => [a.id, a]));

  /** Follows merge_into to its end, guarding against a cycle the model invents. */
  const resolve = (id: number, hops = 0): number => {
    const target = byId.get(id)?.merge_into;
    if (target === null || target === undefined || !byId.has(target) || hops > 8) return id;
    return resolve(target, hops + 1);
  };

  const rows = new Map<string, Record<string, CsvValue>>();
  for (let id = 0; id < clusters.length; id++) {
    const cluster = clusters[id] as NameCluster;
    const root = resolve(id);
    const decided = byId.get(root);
    const name = decided?.name?.trim() || (clusters[root] as NameCluster).provisional;
    const slug = slugify(name);

    const existing = rows.get(slug);
    if (existing) {
      existing.sermon_count = Number(existing.sermon_count) + cluster.count;
      existing.variants = `${existing.variants}|${cluster.members.join("|")}`;
      continue;
    }

    rows.set(slug, {
      slug,
      name,
      kind: kindOf(name) === "ebd" ? kindOf(cluster.provisional) : kindOf(name),
      parent_slug: decided?.parent ? slugify(decided.parent) : null,
      parent_name: decided?.parent ?? null,
      description: decided?.description ?? "",
      sermon_count: cluster.count,
      variants: cluster.members.join("|"),
    });
  }

  // A series that other series point at is the head of a course, whatever its
  // own name looks like: "Confissão de Fé de Westminster" is a real intro
  // lesson AND the parent of the twelve CFW chapters.
  const parents = new Set([...rows.values()].map((r) => r.parent_slug).filter(Boolean));
  for (const row of rows.values()) {
    if (parents.has(row.slug)) row.kind = "cfw";
  }

  const ordered = [...rows.values()].sort(
    (a, b) =>
      Number(b.sermon_count) - Number(a.sermon_count) ||
      String(a.name).localeCompare(String(b.name)),
  );

  await writeFile(
    join(dataDir, "facets/series.csv"),
    writeCsv(
      [
        "slug",
        "name",
        "kind",
        "parent_slug",
        "parent_name",
        "description",
        "sermon_count",
        "variants",
      ],
      ordered,
    ),
  );

  const real = ordered.filter((r) => Number(r.sermon_count) >= 2);
  console.log(`\nwrote data/facets/series.csv`);
  console.log(`  ${ordered.length} series, ${real.length} with 2+ sermons`);
  for (const r of real) {
    console.log(
      `  ${String(r.sermon_count).padStart(3)}  ${r.name}${r.parent_name ? `  (${r.parent_name})` : ""}`,
    );
  }
}

await main();
