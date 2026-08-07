import { describe, expect, it } from "vitest";
import { agreementWith, divergent, jaccard, sameTopics } from "../../src/lib/facets/compare.ts";

/**
 * Whether two labelling configurations disagree, and by how much.
 *
 * Two open questions need answering before anything is reclassified: does
 * reading the whole transcript beat the three-window sample, and does a cheaper
 * model hold up in Portuguese. Neither is answerable by argument, and neither is
 * worth a human reading 40 sermons — so the bench reads all of them and hands
 * back only the rows where the configurations actually differ.
 *
 * Order never carries meaning here: `label-topics` returns topics most-central
 * first, but two configurations picking the same three topics in a different
 * order agree about the sermon.
 */
const row = (sermonId: string, byConfig: Record<string, string[]>) => ({
  sermonId,
  title: `Sermão ${sermonId}`,
  byConfig,
});

describe("sameTopics", () => {
  it("ignores order", () => {
    expect(sameTopics(["graca", "fe"], ["fe", "graca"])).toBe(true);
  });

  it("separates different sets", () => {
    expect(sameTopics(["graca"], ["graca", "fe"])).toBe(false);
  });

  it("counts two empty answers as agreement", () => {
    expect(sameTopics([], [])).toBe(true);
  });
});

describe("jaccard", () => {
  it("is 1 for the same set", () => {
    expect(jaccard(["a", "b"], ["b", "a"])).toBe(1);
  });

  it("is 0 for disjoint sets", () => {
    expect(jaccard(["a"], ["b"])).toBe(0);
  });

  it("measures partial overlap", () => {
    // {a,b} vs {b,c}: one shared of three distinct.
    expect(jaccard(["a", "b"], ["b", "c"])).toBeCloseTo(1 / 3);
  });

  it("is 1 when neither answered, not 0", () => {
    // Both configurations declining to label is agreement, not total
    // disagreement — 0/0 would poison the average.
    expect(jaccard([], [])).toBe(1);
  });
});

describe("divergent", () => {
  const configs = ["A", "B", "C"];

  it("keeps only the rows a human needs to read", () => {
    const rows = [
      row("1", { A: ["graca"], B: ["graca"], C: ["graca"] }),
      row("2", { A: ["graca"], B: ["fe"], C: ["graca"] }),
    ];

    expect(divergent(rows, configs).map((r) => r.sermonId)).toEqual(["2"]);
  });

  it("does not call a reordering a divergence", () => {
    const rows = [row("1", { A: ["graca", "fe"], B: ["fe", "graca"], C: ["graca", "fe"] })];

    expect(divergent(rows, configs)).toEqual([]);
  });

  it("treats a configuration that failed as a divergence", () => {
    // A missing answer is exactly what somebody should look at.
    const rows = [row("1", { A: ["graca"], B: ["graca"] })];

    expect(divergent(rows, configs)).toHaveLength(1);
  });
});

describe("agreementWith", () => {
  it("scores every configuration against the baseline", () => {
    const rows = [
      row("1", { A: ["graca"], B: ["graca"], C: ["fe"] }),
      row("2", { A: ["graca", "fe"], B: ["graca", "fe"], C: ["graca", "fe"] }),
    ];

    expect(agreementWith(rows, "A", ["B", "C"])).toEqual([
      { config: "B", exact: 2, jaccard: 1, total: 2 },
      { config: "C", exact: 1, jaccard: 0.5, total: 2 },
    ]);
  });

  it("leaves the baseline out of its own scoreboard", () => {
    const rows = [row("1", { A: ["graca"], B: ["graca"] })];

    expect(agreementWith(rows, "A", ["A", "B"]).map((a) => a.config)).toEqual(["B"]);
  });

  it("scores a missing answer as total disagreement", () => {
    const rows = [row("1", { A: ["graca"] })];

    expect(agreementWith(rows, "A", ["B"])).toEqual([
      { config: "B", exact: 0, jaccard: 0, total: 1 },
    ]);
  });
});
