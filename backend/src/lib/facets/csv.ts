/**
 * Writes the derived facet tables back out as CSV.
 *
 * The facet files are committed to git and reviewed like the corpus itself, so
 * the output has to be diff-friendly: a stable column order, no quoting unless
 * the value forces it, and `\n` line endings. Quoting only when necessary keeps
 * a one-cell correction to a one-line diff.
 *
 * `parseCsv` in `../corpus.ts` reads this format back.
 */
export type CsvValue = string | number | boolean | null | undefined;

const needsQuotes = (field: string): boolean => /[",\n\r]/.test(field);

const cell = (value: CsvValue): string => {
  if (value === null || value === undefined) return "";
  const field = String(value);
  return needsQuotes(field) ? `"${field.replace(/"/g, '""')}"` : field;
};

export function writeCsv(columns: string[], rows: Record<string, CsvValue>[]): string {
  const lines = [columns.map(cell).join(",")];
  for (const row of rows) {
    lines.push(columns.map((c) => cell(row[c])).join(","));
  }
  return `${lines.join("\n")}\n`;
}
