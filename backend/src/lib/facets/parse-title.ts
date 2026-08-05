import type { BibleBook } from "./bible.ts";
import { parseScriptureRef, type ScriptureRef } from "./parse-scripture.ts";
import { fold } from "./slugify.ts";

/**
 * Reads the facets a sermon title already carries.
 *
 * The corpus has no columns for service type, series or passage — every one of
 * them is latent in the title string, which follows a house grammar that has
 * drifted over six years:
 *
 *   DATE - EBD - <SERIES> - <LESSON>      Sunday school, part of a course
 *   DATE - <PASSAGE> - <THEME>            Sunday preaching
 *   <EVENT>_DATE - <THEME>                conference and congress tracks
 *
 * Nothing here guesses. A title that does not say which series it belongs to
 * gets no series, and the LLM pass in a later phase fills what it can — a wrong
 * facet is worse than a missing one, because it puts a sermon on a page where
 * nobody browsing that page expects it.
 */
export type ServiceType = "culto" | "ebd" | "conferencia" | "congresso" | "confraria" | "diaconia";

export type ParsedTitle = {
  serviceType: ServiceType;
  eventName: string | null;
  seriesCandidate: string | null;
  part: number | null;
  displayTitle: string;
  scripture: ScriptureRef | null;
};

/**
 * The leading date, in every spelling the corpus actually contains.
 *
 * Tolerates the stray space inside "12 -09-2021", the slashes in "30/06/2019",
 * the doubled space in "22-12-2019  - ", and the five-digit year of the
 * "07-05-20223" typo. Eleven rows fail a stricter pattern, and each one would
 * otherwise keep its date glued to the display title.
 */
const LEADING_DATE = /^\s*\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{4,5}\s*-\s*/;

/**
 * "IV Conferência Peregrinos_" or "I Congresso Peregrinos - ".
 *
 * The trailing word is matched as letters only, never `\w`: two rows join the
 * event to the date with an underscore ("Peregrinos_24-09-2022") and `\w` would
 * swallow the day into the event name and leave "09-2022" in the title.
 */
const EVENT_NAME = "[IVX]+\\s+(?:Confer[êe]ncia|Congresso|Confraria)\\s+[A-Za-zÀ-ÿ]+";
const EVENT_PREFIX = new RegExp(`^\\s*(${EVENT_NAME})\\s*[_-]\\s*`, "i");
const EVENT_ANYWHERE = new RegExp(`(${EVENT_NAME})`, "i");
/** The same events when nobody numbered them ("Confraria Peregrinos"). */
const EVENT_KIND = /\b(confer[êe]ncia|congresso|confraria)\b/i;

const ROMAN: Record<string, number> = { i: 1, ii: 2, iii: 3, iv: 4, v: 5, vi: 6 };

const EBD_TOKEN = /^(?:ebd|edb|ed)$/;
const CFW_SEGMENT = /\bCFW\s*(\d{1,2})\b/i;

/** Markers that mean "lesson N of a course", in the order the corpus prefers. */
function readPart(title: string): number | null {
  const parte = title.match(/\bParte\s+([IVX]+|\d{1,3})\b/i);
  if (parte?.[1]) return ROMAN[fold(parte[1])] ?? Number.parseInt(parte[1], 10);

  const aula = title.match(/\bAula\s+(\d{1,3})\b/i);
  if (aula?.[1]) return Number.parseInt(aula[1], 10);

  const pt = title.match(/\(\s*pt\.?\s*(\d{1,3})\s*\)/i);
  if (pt?.[1]) return Number.parseInt(pt[1], 10);

  const paren = title.match(/\((\d{1,3})\)\s*$/);
  if (paren?.[1]) return Number.parseInt(paren[1], 10);

  // A trailing roman numeral, as in "...ensinando na palavra II". Requires a
  // preceding word so a title that simply ends in "I" is not misread.
  const roman = title.match(/\s\w{2,}\s+(I{1,3}|IV|V)\s*$/);
  if (roman?.[1]) return ROMAN[fold(roman[1])] ?? null;

  return null;
}

/**
 * Removes a lesson marker that is glued to the course name.
 *
 * Without this the same course splits into two series: the corpus contains
 * "Igreja e Comunhão" beside "Igreja e Comunhão (2)", and "Permanência no
 * casamento I" beside "...II". Both pairs are one course, and a series index
 * that lists them twice is visibly broken.
 */
function stripPartMarker(segment: string): string {
  return segment
    .replace(/\s*\b(?:Aula|Parte)\s+(?:[IVX]+|\d{1,3})\s*$/i, "")
    .replace(/\s*\(\s*(?:pt\.?\s*)?\d{1,3}\s*\)\s*$/i, "")
    .replace(/(\s\w{2,})\s+(?:I{1,3}|IV|V)\s*$/, "$1")
    .trim();
}

/**
 * True when a segment is nothing but a passage.
 *
 * "Atos 27" is a reference and must not become a series; "O Livro dos Reis" is
 * a seventeen-part course that merely contains a book word. The test is how
 * much of the segment the reference accounts for.
 */
function isBareReference(books: BibleBook[], segment: string): boolean {
  const ref = parseScriptureRef(books, segment);
  if (!ref) return false;
  const leftover = fold(segment)
    .replace(/[0-9.,:;–—-]/g, " ")
    .replace(/\b(?:e|de|do|da|dos|das|a|o)\b/g, " ")
    .trim();
  // Whatever is left once the digits and the book name are removed.
  const bookWords = fold(ref.bookSlug.replace(/-/g, " "));
  const rest = leftover
    .split(/\s+/)
    .filter((w) => w.length > 0 && !bookWords.includes(w))
    .join(" ");
  return rest.length <= 3;
}

function readSeries(
  books: BibleBook[],
  segments: string[],
  serviceType: ServiceType,
  eventName: string | null,
): string | null {
  const cfw = segments.map((s) => s.match(CFW_SEGMENT)).find((m) => m !== null);
  if (cfw?.[1]) return `CFW ${Number.parseInt(cfw[1], 10)}`;

  if (serviceType === "conferencia" || serviceType === "congresso" || serviceType === "confraria") {
    return eventName;
  }

  if (serviceType === "diaconia") return "Diaconia";

  if (serviceType !== "ebd") return null;

  const ebdAt = segments.findIndex((s) => EBD_TOKEN.test(fold(s)));
  const next = segments[ebdAt + 1];
  if (ebdAt === -1 || next === undefined) return null;

  // "Diaconia Aula 2" names the course and the lesson in one segment.
  const name = stripPartMarker(next);
  if (!name || isBareReference(books, name)) return null;
  return name;
}

export function parseTitle(books: BibleBook[], title: string, description?: string): ParsedTitle {
  const eventInTitle = title.match(EVENT_PREFIX) ?? title.match(EVENT_ANYWHERE);
  const eventInDescription = description ? EVENT_ANYWHERE.exec(description) : null;
  const eventName = (eventInTitle?.[1] ?? eventInDescription?.[1])?.trim() ?? null;

  const displayTitle = title.replace(EVENT_PREFIX, "").replace(LEADING_DATE, "").trim();

  const segments = displayTitle
    .split(/\s+-\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  // The event may be named in the title, in the description, or only by the
  // bare word when nobody numbered that year's edition.
  const folded = `${fold(eventName ?? "")} ${fold(title)}`;
  const kind = folded.match(EVENT_KIND)?.[1];

  let serviceType: ServiceType = "culto";
  if (segments.some((s) => EBD_TOKEN.test(fold(s)))) serviceType = "ebd";
  else if (kind?.startsWith("confer")) serviceType = "conferencia";
  else if (kind === "congresso") serviceType = "congresso";
  else if (kind === "confraria") serviceType = "confraria";
  else if (/\bdiaconia\b/.test(folded)) serviceType = "diaconia";

  return {
    serviceType,
    eventName,
    seriesCandidate: readSeries(books, segments, serviceType, eventName),
    part: readPart(displayTitle),
    displayTitle,
    scripture: parseScriptureRef(books, displayTitle),
  };
}
