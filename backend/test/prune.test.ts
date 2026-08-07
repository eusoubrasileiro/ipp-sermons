import { describe, expect, it } from "vitest";
import { MAX_PRUNE_FRACTION, prunePlan } from "../src/lib/prune.ts";

const ids = (n: number, prefix = "s") => Array.from({ length: n }, (_, i) => `${prefix}${i}`);

describe("prunePlan", () => {
  it("removes the sermons the corpus no longer offers", () => {
    // The nine truncated downloads: still in Postgres, gone from the CSV.
    const indexed = ids(503);
    const corpus = indexed.slice(0, 494);
    expect(prunePlan(indexed, corpus)).toEqual(ids(503).slice(494));
  });

  it("removes nothing when the corpus still covers everything indexed", () => {
    expect(prunePlan(ids(500), ids(500))).toEqual([]);
  });

  it("ignores sermons the corpus adds but the database has not seen", () => {
    // A normal corpus update: new rows are the indexer's job, not the pruner's.
    expect(prunePlan(ids(400), ids(450))).toEqual([]);
  });

  it("removes nothing on a first run against an empty database", () => {
    expect(prunePlan([], ids(500))).toEqual([]);
  });

  it("refuses to prune more than the ceiling in one run", () => {
    // The guard that matters. `loadSermons` gained a filter today whose first
    // draft rejected every row on a missing column; without this, that bug
    // deletes the production corpus instead of skipping nine sermons.
    const indexed = ids(500);
    expect(() => prunePlan(indexed, indexed.slice(0, 400))).toThrow(/20\.0%/);
    expect(() => prunePlan(indexed, indexed.slice(0, 400))).toThrow(/refusing/i);
  });

  it("refuses an empty corpus outright", () => {
    expect(() => prunePlan(ids(500), [])).toThrow(/refusing/i);
  });

  it("accepts a run exactly at the ceiling", () => {
    const indexed = ids(1000);
    const keep = 1000 - MAX_PRUNE_FRACTION * 1000;
    expect(prunePlan(indexed, indexed.slice(0, keep))).toHaveLength(MAX_PRUNE_FRACTION * 1000);
  });
});
