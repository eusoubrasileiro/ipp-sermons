import type { SearchResult } from "@ipp/shared";
import { describe, expect, it } from "vitest";
import { groupBySermon } from "./group.ts";

const hit = (id: string, chunkIndex: number): SearchResult => ({
  id,
  title: `sermão ${id}`,
  artist: "Reverendo Bruno Melo",
  date: "2021-07-18",
  durationStr: "41:00",
  soundcloudUrl: null,
  spotifyUrl: null,
  content: `trecho ${chunkIndex}`,
  score: 1,
  chunkIndex,
});

describe("groupBySermon", () => {
  it("collapses several chunks of one sermon into a single card", () => {
    const groups = groupBySermon([hit("a", 1), hit("a", 7), hit("b", 2)]);
    expect(groups).toHaveLength(2);
    expect(groups[0]?.top.chunkIndex).toBe(1);
    expect(groups[0]?.more.map((m) => m.chunkIndex)).toEqual([7]);
    expect(groups[1]?.more).toEqual([]);
  });

  it("keeps the API's ranking order", () => {
    const groups = groupBySermon([hit("b", 0), hit("a", 0), hit("b", 5)]);
    expect(groups.map((g) => g.top.id)).toEqual(["b", "a"]);
  });

  it("handles an empty result set", () => {
    expect(groupBySermon([])).toEqual([]);
  });
});
