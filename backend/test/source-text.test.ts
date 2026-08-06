import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * A source file that git reads as binary is a source file nobody can review.
 *
 * `topics.ts` used a raw NUL as a composite-key separator — correct at
 * runtime, and it made `git diff` print "Binary files differ" instead of the
 * code. The pre-push reviewer blocked on it, which is the only reason it was
 * caught: no test, no linter and no typechecker cares about a NUL inside a
 * template literal. Writing it as an escape keeps the file text and the
 * runtime behaviour identical.
 */
const SRC = join(import.meta.dirname, "../src");

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return entry.name.endsWith(".ts") ? [path] : [];
  });
}

describe("backend sources", () => {
  it("contain no byte that makes git treat them as binary", () => {
    const binary = sourceFiles(SRC).filter((path) => readFileSync(path).includes(0));
    expect(binary).toEqual([]);
  });

  it("finds the sources it is meant to be scanning", () => {
    expect(sourceFiles(SRC).length).toBeGreaterThan(20);
  });
});
