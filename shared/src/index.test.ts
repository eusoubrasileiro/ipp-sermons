import { describe, expect, it } from "vitest";
import { SearchRequestSchema, SearchResultSchema, SuggestionRequestSchema } from "./index.ts";

describe("SearchRequestSchema", () => {
  it("defaults limit to 10", () => {
    expect(SearchRequestSchema.parse({ query: "graça" }).limit).toBe(10);
  });

  it("trims and rejects too-short queries", () => {
    expect(SearchRequestSchema.safeParse({ query: " a " }).success).toBe(false);
    expect(SearchRequestSchema.parse({ query: "  fé  " }).query).toBe("fé");
  });

  it("caps limit at 50", () => {
    expect(SearchRequestSchema.safeParse({ query: "fé", limit: 51 }).success).toBe(false);
  });
});

describe("SearchResultSchema", () => {
  it("allows null playback urls", () => {
    const r = SearchResultSchema.parse({
      id: "1",
      title: "Tito 2",
      artist: "Rev. Bruno",
      date: "2020-05-03",
      durationStr: "45:49",
      soundcloudUrl: null,
      spotifyUrl: null,
      content: "texto",
      score: 0.03,
      chunkIndex: 0,
    });
    expect(r.soundcloudUrl).toBeNull();
  });
});

describe("SuggestionRequestSchema", () => {
  it("requires a non-trivial suggestion", () => {
    expect(SuggestionRequestSchema.safeParse({ suggestion: "a" }).success).toBe(false);
    expect(SuggestionRequestSchema.safeParse({ suggestion: "falta um sermão" }).success).toBe(true);
  });
});
