import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { loadSermons } from "../lib/corpus.ts";
import { parseCsv } from "../lib/csv.ts";
import { DATA_DIR } from "../lib/data-dir.ts";
import { writeCsv } from "../lib/facets/csv.ts";
import { TAXONOMY_COLUMNS, type TaxonomyProposal, taxonomyRows } from "../lib/facets/topics.ts";
import { createOpenRouterLlm, LLM_MODEL_STRONG } from "../lib/llm.ts";
import { createUsageMeter } from "../lib/usage.ts";

/**
 * Proposes the topic taxonomy from the corpus itself.
 *
 *   pnpm --filter @ipp/backend propose:taxonomy [--dry-run]
 *
 * Deliberately not imported from Desiring God. Their 300-leaf taxonomy was
 * built for 17.000 resources; over 456 sermons most of it would be empty, and
 * an empty facet is worse than a missing one -- it promises something the
 * archive does not have. So the groups and leaves are derived from what this
 * church has actually preached.
 *
 * One call over every title, series and passage in the corpus: a few thousand
 * tokens, so the stronger model costs cents and the output is committed as
 * ground truth. `label-topics.ts` may then only pick from it.
 */

/** What this run spends, reported by OpenRouter rather than estimated. */
const meter = createUsageMeter();
const DRY_RUN = process.argv.includes("--dry-run");

const SCHEMA = {
  type: "object",
  properties: {
    topicos: {
      type: "array",
      items: {
        type: "object",
        properties: {
          grupo: { type: "string" },
          topico: { type: "string" },
          descricao: { type: "string" },
        },
        required: ["grupo", "topico", "descricao"],
        additionalProperties: false,
      },
    },
  },
  required: ["topicos"],
  additionalProperties: false,
} as const;

const SYSTEM = `Você organiza o acervo de uma igreja presbiteriana brasileira (Igreja Presbiteriana Peregrinos) em uma taxonomia de temas para navegação no site.

Receberá a lista completa de sermões do acervo: título, série e passagem bíblica quando houver.

Proponha uma taxonomia de DOIS NÍVEIS: grupo -> tópico.

- Entre 7 e 9 grupos, largos e reconhecíveis por um membro comum (ex.: "Vida Cristã", "Cristo e o Evangelho", "Igreja e Ministério", "Família e Relacionamentos", "Sofrimento e Esperança", "Bíblia e Doutrina").
- Entre 55 e 70 tópicos no total, distribuídos entre os grupos.
- Cada tópico precisa ter pelo menos 2 ou 3 sermões PLAUSÍVEIS neste acervo. Um tópico que ficaria vazio é pior do que não existir.
- Nomes em português do Brasil, curtos e concretos ("Ansiedade e medo", "Batismo", "Justificação pela fé"), não abstratos ("Questões diversas").
- descricao: uma frase de até 120 caracteres dizendo o que cabe no tópico.

Regras rígidas:
- Cubra o que este acervo realmente prega, incluindo a forte presença de catecismo, Confissão de Fé de Westminster e governo eclesiástico.
- Não crie tópicos que sejam livros da Bíblia nem nomes de pregadores: isso já é outra faceta.
- Não repita o mesmo tópico em dois grupos.`;

async function main(): Promise<void> {
  const csvText = await readFile(join(DATA_DIR, "metadata.csv"), "utf8");
  const { sermons } = loadSermons(csvText);

  const facets = new Map(
    parseCsv(await readFile(join(DATA_DIR, "facets/sermon_facets.csv"), "utf8")).map((r) => [
      r.sermon_id,
      r,
    ]),
  );

  const scriptures = new Map<string, string>();
  for (const row of parseCsv(
    await readFile(join(DATA_DIR, "facets/sermon_scriptures.csv"), "utf8"),
  )) {
    if (row.sermon_id && !scriptures.has(row.sermon_id)) {
      scriptures.set(row.sermon_id, `${row.book_slug} ${row.chapter !== "0" ? row.chapter : ""}`);
    }
  }

  const lines = sermons.map((s) => {
    const facet = facets.get(s.id);
    const parts = [facet?.display_title || s.title];
    if (facet?.series_candidate) parts.push(`série: ${facet.series_candidate}`);
    const ref = scriptures.get(s.id);
    if (ref) parts.push(`texto: ${ref.trim()}`);
    return `- ${parts.join(" | ")}`;
  });

  console.log(`${sermons.length} sermons in the prompt  (${LLM_MODEL_STRONG})`);

  if (DRY_RUN) {
    console.log(lines.slice(0, 15).join("\n"));
    console.log("\n--dry-run: stopping before the LLM call.");
    return;
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is not set");

  const llm = createOpenRouterLlm({ apiKey, model: LLM_MODEL_STRONG, onUsage: meter.record });
  const answer = await llm.complete<{ topicos: TaxonomyProposal[] }>({
    system: SYSTEM,
    user: lines.join("\n"),
    schema: SCHEMA,
    schemaName: "taxonomia_temas",
    maxTokens: 8000,
  });

  const rows = taxonomyRows(answer.topicos);
  await writeFile(join(DATA_DIR, "facets/taxonomy.csv"), writeCsv(TAXONOMY_COLUMNS, rows));

  const byGroup = new Map<string, string[]>();
  for (const r of rows) {
    const group = String(r.grupo_nome);
    byGroup.set(group, [...(byGroup.get(group) ?? []), String(r.topico_nome)]);
  }

  console.log(`\nwrote data/facets/taxonomy.csv — ${byGroup.size} groups, ${rows.length} topics\n`);
  for (const [group, topics] of byGroup) {
    console.log(`  ${group}  (${topics.length})`);
    console.log(`      ${topics.join(" · ")}`);
  }
}

await main();

// Reported by OpenRouter, not multiplied out of a price list: the list goes
// stale without anyone noticing. Silent when the run made no paid call.
const spend = meter.summary();
if (spend) console.log(`\n${spend}`);
