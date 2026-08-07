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

/**
 * Guards a load that filters rows against known keys.
 *
 * A file whose rows *all* filter out is a column-name or slug mismatch, not an
 * empty corpus, and it is invisible: the load reports zero, the deploy
 * succeeds, and every facet page renders and lists nothing. This turns that
 * into a failure at the point it can still be fixed.
 */
export function assertMatched(expected: string, rows: number, matched: number): void {
  if (rows > 0 && matched === 0) {
    throw new Error(`${rows} rows read but none matched a known key — expected ${expected}`);
  }
}

/** A CSV integer cell, or null when the column is blank — a whole-chapter reference. */
export function int(value: string | undefined): number | null {
  const n = Number.parseInt((value ?? "").trim(), 10);
  return Number.isFinite(n) ? n : null;
}
