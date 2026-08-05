import type { BibleBook } from "./bible.ts";
import type { CsvValue } from "./csv.ts";
import { chaptersOf } from "./parse-scripture.ts";

/**
 * The scripture pass over sermons whose title names no book.
 *
 * 73% of the corpus announces its passage in the title and is parsed for free.
 * The rest opens with the preacher reading it aloud instead, which is what this
 * asks a model to find.
 *
 * Measured on this corpus before choosing the window: 400 words recovers 53% of
 * the title-less sermons, 1000 recovers 78%, 2500 recovers ~97%. Past that the
 * mentions are supporting citations rather than the text being preached, so a
 * longer window buys hallucination, not coverage.
 */
export const WINDOW_WORDS = 2500;

export type ScriptureDecision = {
  livro: string | null;
  capitulo_inicio: number | null;
  versiculo_inicio: number | null;
  capitulo_fim: number | null;
  versiculo_fim: number | null;
  /** Kept in the cache so a human can spot-check without re-reading transcripts. */
  justificativa: string;
};

/** The opening of a transcript, whitespace collapsed. */
export function openingWords(text: string, limit = WINDOW_WORDS): string {
  return text.split(/\s+/).filter(Boolean).slice(0, limit).join(" ");
}

/** The closed list the model chooses from — an invented book cannot be returned. */
export const bookNames = (books: BibleBook[]): string[] => books.map((b) => b.name);

export function buildSchema(books: BibleBook[]): unknown {
  return {
    type: "object",
    properties: {
      livro: { type: ["string", "null"], enum: [...bookNames(books), null] },
      capitulo_inicio: { type: ["integer", "null"] },
      versiculo_inicio: { type: ["integer", "null"] },
      capitulo_fim: { type: ["integer", "null"] },
      versiculo_fim: { type: ["integer", "null"] },
      justificativa: { type: "string" },
    },
    required: [
      "livro",
      "capitulo_inicio",
      "versiculo_inicio",
      "capitulo_fim",
      "versiculo_fim",
      "justificativa",
    ],
    additionalProperties: false,
  };
}

/**
 * The instruction, and the whole reason this pass is trustworthy.
 *
 * `null` is not a fallback here, it is the expected answer for a large part of
 * the remaining corpus: catechism lessons on the Westminster Confession, the
 * Ten Commandments, apologetics and conference talks have no primary passage.
 * A model pushed to always answer files them under whatever verse it saw first,
 * and the Bible index is then wrong precisely where it should be empty.
 */
export const EXTRACT_SYSTEM = `Você lê o início da transcrição de um sermão da Igreja Presbiteriana Peregrinos (Brasil) e identifica QUAL PASSAGEM BÍBLICA está sendo pregada.

Devolva o livro, capítulo e versículos do texto principal — aquele que o pregador anuncia, lê em voz alta ou diz que vai expor.

Devolva livro = null quando o sermão NÃO tem uma passagem principal. Isso é comum e esperado:
- aulas de catecismo ou da Confissão de Fé de Westminster;
- séries sobre os Dez Mandamentos;
- apologética, missões, temas doutrinários gerais;
- palestras de congresso e conferência.

Regras rígidas:
- NÃO escolha uma passagem só porque ela foi citada de passagem. O texto principal é lido, anunciado ou exposto.
- Na dúvida entre uma citação incidental e nenhuma passagem, devolva null.
- Se o pregador anuncia um capítulo inteiro, deixe os versículos nulos.
- Se anuncia um intervalo de capítulos, use capitulo_inicio e capitulo_fim.
- justificativa: uma frase curta em português citando o que no texto indicou a resposta.`;

export function buildPrompt(title: string, transcript: string): string {
  return `Título: ${title}\n\nInício da transcrição:\n${openingWords(transcript)}`;
}

/**
 * The model's answer as CSV rows, or none at all.
 *
 * Every rejection here is a hallucination that would otherwise reach the index
 * silently: an unknown book, or a chapter the book does not have. "Naum 12"
 * builds a page nobody can explain, and no test downstream would catch it.
 */
export function decisionToRows(
  books: BibleBook[],
  sermonId: string,
  decision: ScriptureDecision,
): Record<string, CsvValue>[] {
  if (!decision.livro) return [];

  const book = books.find((b) => b.name === decision.livro);
  if (!book) return [];

  const start = decision.capitulo_inicio;
  if (start !== null && (start < 1 || start > book.chapters)) return [];

  const row = (chapter: number, verseStart: number | null, verseEnd: number | null) => ({
    sermon_id: sermonId,
    book_slug: book.slug,
    chapter,
    verse_start: verseStart,
    verse_end: verseEnd,
    source: "transcricao",
    is_primary: true,
  });

  // A book with no chapter is still a usable facet; 0 keeps the row's identity
  // complete, exactly as the title parser writes it.
  if (start === null) return [row(0, null, null)];

  const end = Math.min(decision.capitulo_fim ?? start, book.chapters);
  const chapters = chaptersOf({ chapterStart: start, chapterEnd: end });

  return chapters.map((chapter) =>
    row(
      chapter,
      chapter === start ? decision.versiculo_inicio : null,
      chapter === end ? decision.versiculo_fim : null,
    ),
  );
}
