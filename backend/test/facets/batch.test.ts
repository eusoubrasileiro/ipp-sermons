import { describe, expect, it, vi } from "vitest";
import { cliLimit, reportFailures, runBatch } from "../../src/lib/facets/batch.ts";

/**
 * The loop both paid passes run on.
 *
 * The behaviour that matters is what happens when one call fails: the run must
 * finish the rest and report, because aborting halfway through 456 LLM calls
 * wastes the ones already made.
 */
describe("cliLimit", () => {
  it("reads --limit", () => {
    expect(cliLimit(["node", "script.ts", "--limit", "5"])).toBe(5);
  });

  it("is unbounded without one, and ignores a non-number", () => {
    expect(cliLimit(["node", "script.ts"])).toBe(Number.POSITIVE_INFINITY);
    expect(cliLimit(["node", "script.ts", "--limit", "todos"])).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("runBatch", () => {
  const label = (n: number) => `item ${n}`;

  it("runs every item and reports no failures", async () => {
    const seen: number[] = [];
    const failures = await runBatch(
      [1, 2, 3, 4, 5],
      async (n) => {
        seen.push(n);
      },
      { label, concurrency: 2 },
    );

    expect(seen.sort()).toEqual([1, 2, 3, 4, 5]);
    expect(failures).toEqual([]);
  });

  it("keeps going after one item throws", async () => {
    const seen: number[] = [];
    const failures = await runBatch(
      [1, 2, 3],
      async (n) => {
        if (n === 2) throw new Error("sem resposta");
        seen.push(n);
      },
      { label, concurrency: 1 },
    );

    expect(seen).toEqual([1, 3]);
    expect(failures).toEqual(["item 2: sem resposta"]);
  });

  it("flushes progress before the end so an interrupted run keeps its work", async () => {
    const flushes: number[] = [];
    await runBatch([1, 2, 3, 4, 5, 6], async () => {}, {
      label,
      concurrency: 2,
      flushEvery: 4,
      onFlush: async (done) => {
        flushes.push(done);
      },
    });

    expect(flushes.length).toBeGreaterThan(0);
    expect(flushes.at(-1)).toBeLessThanOrEqual(6);
  });
});

describe("reportFailures", () => {
  it("says nothing when nothing failed", () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    reportFailures([]);
    expect(log).not.toHaveBeenCalled();
    log.mockRestore();
  });

  it("names the failures without printing hundreds of them", () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    reportFailures(Array.from({ length: 50 }, (_, i) => `item ${i}: erro`));

    // One heading plus at most ten lines.
    expect(log.mock.calls.length).toBeLessThanOrEqual(11);
    expect(String(log.mock.calls[0]?.[0])).toContain("50");
    log.mockRestore();
  });
});
