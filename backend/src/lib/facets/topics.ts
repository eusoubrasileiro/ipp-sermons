import type { CsvValue } from "./csv.ts";
import { slugify } from "./slugify.ts";

/**
 * The topic taxonomy, and the labels sermons carry against it.
 *
 * Two levels -- group then topic -- because Desiring God's own /topics page is
 * two levels ("The Christian Life > Life Issues > Anger") and one structure
 * doing the work of their /topics and /themes is right for 456 sermons. Three
 * would be ceremony.
 *
 * The taxonomy is proposed once from the real corpus, committed, and then
 * closed: labelling picks from it and may not extend it. A topic invented
 * during labelling would either break the foreign key at load time or, worse,
 * build a facet page that renders and lists nothing.
 */
export const TAXONOMY_COLUMNS = [
  "grupo_slug",
  "grupo_nome",
  "topico_slug",
  "topico_nome",
  "descricao",
];

/** Portuguese here because `index-facets.ts` and taxonomy.csv already use these names. */
export const SERMON_TOPIC_COLUMNS = ["sermon_id", "topico_slug", "confianca"];

/**
 * At most four. A sermon tagged with everything it mentions is tagged with
 * nothing useful: every facet page then fills with sermons that only glance
 * at it, which is the failure mode of an auto-labelled archive.
 */
export const MAX_TOPICS_PER_SERMON = 4;

export type TaxonomyProposal = { grupo: string; topico: string; descricao: string };
export type TopicLabel = { topico_slug: string; confianca: number };

/**
 * A sermon reduced to what the labeller needs to read.
 *
 * Not the opening alone: a sermon announces its text in the first minutes but
 * reaches its application -- marriage, money, anxiety, the topics a visitor
 * actually searches for -- in the second half. Not the whole transcript
 * either, which doubles the spend for the least informative words. Three
 * windows cover announcement, argument and application.
 */
export function sampleTranscript(text: string, head = 1500, middle = 1500, tail = 800): string {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length <= head + middle + tail) return words.join(" ");

  const midStart = Math.floor((words.length - middle) / 2);
  return [
    words.slice(0, head).join(" "),
    words.slice(midStart, midStart + middle).join(" "),
    words.slice(-tail).join(" "),
  ].join("\n[…]\n");
}

export function taxonomyRows(proposal: TaxonomyProposal[]): Record<string, CsvValue>[] {
  const rows: Record<string, CsvValue>[] = [];
  const seenPair = new Set<string>();
  const seenSlug = new Set<string>();

  for (const entry of proposal) {
    const grupo = entry.grupo?.trim() ?? "";
    const topico = entry.topico?.trim() ?? "";
    if (!grupo || !topico) continue;

    // Escaped, not a literal NUL: an unescaped one makes git read this
    // source file as binary, so no reviewer or diff can see it again.
    const pair = `${grupo}\u0000${topico}`;
    if (seenPair.has(pair)) continue;
    seenPair.add(pair);

    // Two groups may legitimately name a leaf the same thing; the slug is the
    // key, so the second one has to differ.
    const base = slugify(topico);
    let slug = base;
    for (let n = 2; seenSlug.has(slug); n += 1) slug = `${base}-${n}`;
    seenSlug.add(slug);

    rows.push({
      grupo_slug: slugify(grupo),
      grupo_nome: grupo,
      topico_slug: slug,
      topico_nome: topico,
      descricao: entry.descricao?.trim() ?? "",
    });
  }

  return rows;
}

const clamp = (n: number): number => Math.min(1, Math.max(0, Number.isFinite(n) ? n : 0));

export function labelRows(
  known: Set<string>,
  sermonId: string,
  labels: TopicLabel[],
): Record<string, CsvValue>[] {
  const seen = new Set<string>();
  const rows: Record<string, CsvValue>[] = [];

  for (const label of labels) {
    if (!known.has(label.topico_slug) || seen.has(label.topico_slug)) continue;
    seen.add(label.topico_slug);
    rows.push({
      sermon_id: sermonId,
      topico_slug: label.topico_slug,
      confianca: clamp(label.confianca),
    });
    if (rows.length === MAX_TOPICS_PER_SERMON) break;
  }

  return rows;
}

/**
 * The labelling instruction, shared so that every caller asks the same question.
 *
 * `compare-topics` exists to tell one configuration from another; if it also
 * varied the prompt it would not be comparing anything. Keeping the prompt here
 * rather than inside the script makes that structural instead of a discipline
 * somebody has to remember.
 */
export const TOPICS_SYSTEM = `Você classifica sermões de uma igreja presbiteriana brasileira (Igreja Presbiteriana Peregrinos) numa taxonomia FECHADA de temas.

Receberá o título, a passagem bíblica quando houver, e trechos da transcrição (início, meio e fim).

Devolva de 1 a ${MAX_TOPICS_PER_SERMON} tópicos, do mais central para o menos central, escolhidos EXCLUSIVAMENTE da lista fornecida.

- confianca: 1.0 quando o tópico é o assunto do sermão; 0.5 quando é tratado de forma relevante mas secundária; abaixo de 0.4 não devolva.
- Prefira poucos tópicos certos a muitos aproximados. Um sermão marcado com tudo que menciona não ajuda ninguém a encontrá-lo.
- Não escolha um tópico só porque a palavra aparece: o sermão precisa tratar do assunto.`;

/**
 * The reply schema, with the taxonomy as a closed enum.
 *
 * The enum is the whole guarantee: with `strict: true` the provider constrains
 * decoding, so a topic outside the taxonomy is unreachable rather than merely
 * discouraged. A model that only supports `response_format` cannot offer this.
 */
export function topicsSchema(known: Set<string>): unknown {
  return {
    type: "object",
    properties: {
      topicos: {
        type: "array",
        items: {
          type: "object",
          properties: {
            topico_slug: { type: "string", enum: [...known] },
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
}

/** What the model reads about one sermon. `body` is the sample or the whole transcript. */
export function topicsPrompt(title: string, reference: string | undefined, body: string): string {
  return `Título: ${title}${reference ? `\nPassagem: ${reference}` : ""}\n\nTranscrição:\n${body}`;
}
