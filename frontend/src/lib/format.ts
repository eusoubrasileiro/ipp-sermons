/** Display helpers for sermon metadata. All output is pt-BR. */

const MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"];

/**
 * Corpus titles carry the preaching date as a prefix ("18-07-2021 - Gênesis
 * 17.9-27", sometimes with slashes). The card shows the date in its own field,
 * so repeating it in the heading only costs line width on a phone.
 */
export function stripLeadingDate(title: string): string {
  return title.replace(/^\s*\d{1,2}[-/]?\d{2}[-/]?\d{2,4}\s*[-–—]\s*/, "").trim() || title.trim();
}

/** "2021-07-18" -> "18 jul 2021". Short enough to sit inline on a phone. */
export function formatDate(iso: string): string {
  const [year, month, day] = iso.split("-");
  const name = MONTHS[Number(month) - 1];
  if (!year || !day || !name) return iso;
  return `${day} ${name} ${year}`;
}

/** Drops a leading "0:" so "0:45:49" reads as "45:49". */
export function formatDuration(durationStr: string): string {
  return durationStr.replace(/^0:/, "");
}
