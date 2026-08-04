import { describe, expect, it } from "vitest";
import { chunkHash, chunkText, loadSermons, parseCsv, resolveDate } from "../src/lib/corpus.ts";
import { normalize, toVectorLiteral } from "../src/lib/embeddings.ts";

describe("parseCsv", () => {
  it("parses a simple table", () => {
    const rows = parseCsv("a,b\n1,2\n3,4\n");
    expect(rows).toEqual([
      { a: "1", b: "2" },
      { a: "3", b: "4" },
    ]);
  });

  it("keeps commas inside quoted fields", () => {
    // Sermon descriptions routinely contain commas.
    const rows = parseCsv('name,description\nTito 2,"Graça, fé e obras"\n');
    expect(rows[0]?.description).toBe("Graça, fé e obras");
  });

  it("handles escaped quotes and embedded newlines", () => {
    const rows = parseCsv('name,description\nX,"diz ""paz"" a\ntodos"\n');
    expect(rows[0]?.description).toBe('diz "paz" a\ntodos');
  });

  it("ignores blank trailing lines", () => {
    expect(parseCsv("a,b\n1,2\n\n")).toHaveLength(1);
  });
});

describe("resolveDate", () => {
  it("accepts a well-formed ISO date", () => {
    expect(resolveDate("2021-09-17", "")?.toISOString()).toBe("2021-09-17T00:00:00.000Z");
  });

  it("falls back to the unix timestamp when the date is empty", () => {
    // Five conference sermons predate the date column but carry a timestamp.
    expect(resolveDate("", "1568514829")?.getUTCFullYear()).toBe(2019);
  });

  it("rejects the corrupt year and falls back", () => {
    // One row reads "0223-05-07", from a typo'd "07-05-20223" title.
    const d = resolveDate("0223-05-07", "1684690254");
    expect(d?.getUTCFullYear()).toBe(2023);
  });

  it("returns null when neither source is usable", () => {
    expect(resolveDate("", "")).toBeNull();
    expect(resolveDate("not-a-date", "0")).toBeNull();
  });
});

describe("chunkText", () => {
  const words = (n: number) => Array.from({ length: n }, (_, i) => `w${i}`).join(" ");

  it("returns a single chunk for short text", () => {
    expect(chunkText(words(50))).toHaveLength(1);
  });

  it("overlaps consecutive chunks", () => {
    const chunks = chunkText(words(400), 200, 30);
    const first = chunks[0]?.split(" ") ?? [];
    const second = chunks[1]?.split(" ") ?? [];
    // The stride is 170, so chunk 2 starts at w170 -- inside chunk 1.
    expect(first).toContain("w170");
    expect(second[0]).toBe("w170");
  });

  it("covers the whole transcript", () => {
    const chunks = chunkText(words(1000), 200, 30);
    expect(chunks.at(-1)).toContain("w999");
  });

  it("handles empty input", () => {
    expect(chunkText("")).toEqual([]);
    expect(chunkText("   ")).toEqual([]);
  });
});

describe("chunkHash", () => {
  it("is stable for identical content", () => {
    expect(chunkHash("s1", 0, "texto")).toBe(chunkHash("s1", 0, "texto"));
  });

  it("changes when content, index or sermon changes", () => {
    const base = chunkHash("s1", 0, "texto");
    expect(chunkHash("s1", 0, "outro")).not.toBe(base);
    expect(chunkHash("s1", 1, "texto")).not.toBe(base);
    expect(chunkHash("s2", 0, "texto")).not.toBe(base);
  });
});

describe("normalize", () => {
  it("rescales to unit length", () => {
    const v = normalize([3, 4]);
    expect(Math.hypot(...v)).toBeCloseTo(1, 10);
  });

  it("fixes the 0.697-norm vectors the API actually returns", () => {
    // Truncated Matryoshka output is not unit length; pgvector cosine assumes
    // it is. This is the bug that would silently skew every similarity.
    const truncated = [0.4, 0.5, 0.2];
    const norm = Math.hypot(...normalize(truncated));
    expect(Math.hypot(...truncated)).toBeLessThan(1);
    expect(norm).toBeCloseTo(1, 10);
  });

  it("leaves a zero vector alone rather than dividing by zero", () => {
    expect(normalize([0, 0])).toEqual([0, 0]);
  });
});

describe("toVectorLiteral", () => {
  it("formats for pgvector", () => {
    expect(toVectorLiteral([0.1, -0.2])).toBe("[0.100000,-0.200000]");
  });
});

describe("loadSermons", () => {
  const header =
    "name,processed,txt,artist,duration_str,id,duration,timestamp,sc_suffix_url,sp_suffix_url,date,words,sentences,words_min,sentences_min,score";
  const row = (over: Record<string, string> = {}) => {
    const base: Record<string, string> = {
      name: "Tito 2",
      processed: "True",
      txt: "Tito 2.txt",
      artist: "Reverendo Bruno Melo",
      duration_str: "45:49",
      id: "123",
      duration: "2749.5",
      timestamp: "1588550000",
      sc_suffix_url: "tito-2",
      sp_suffix_url: "abc",
      date: "2020-05-03",
      words: "6412",
      sentences: "411",
      words_min: "102.3",
      sentences_min: "6.5",
      score: "82.0",
      ...over,
    };
    return header
      .split(",")
      .map((h) => base[h] ?? "")
      .join(",");
  };

  it("loads an eligible sermon", () => {
    const { sermons } = loadSermons(`${header}\n${row()}\n`);
    expect(sermons).toHaveLength(1);
    expect(sermons[0]?.artist).toBe("Reverendo Bruno Melo");
    expect(sermons[0]?.durationSec).toBe(2750);
  });

  it("does not append a second .txt extension", () => {
    // The CSV column already ends in .txt; doubling it finds zero files.
    const { sermons } = loadSermons(`${header}\n${row()}\n`);
    expect(sermons[0]?.transcriptFile).toBe("Tito 2.txt");
  });

  it("skips unprocessed rows and rows at or below the score cutoff", () => {
    const csv = [
      header,
      row({ processed: "False" }),
      row({ score: "17.3" }),
      row({ score: "50" }),
    ].join("\n");
    const { sermons, skipped } = loadSermons(csv);
    expect(sermons).toHaveLength(0);
    expect(skipped).toHaveLength(3);
  });

  it("keeps a sermon whose date is recoverable from the timestamp", () => {
    const { sermons } = loadSermons(`${header}\n${row({ date: "" })}\n`);
    expect(sermons).toHaveLength(1);
    expect(sermons[0]?.date.getUTCFullYear()).toBe(2020);
  });

  it("falls back to the name when there is no SoundCloud id", () => {
    const { sermons } = loadSermons(`${header}\n${row({ id: "" })}\n`);
    expect(sermons[0]?.id).toBe("Tito 2");
  });
});

describe("loadSermons deduplication", () => {
  const header = "name,processed,txt,artist,duration_str,id,duration,timestamp,date,score";
  const row = (id: string, score: string, txt: string) =>
    `Nono mandamento,True,${txt},Pastor X,30:00,${id},1800,1602000000,2020-10-11,${score}`;

  it("keeps only one row per transcript file", () => {
    // One sermon was uploaded to SoundCloud twice under different ids.
    const csv = [
      header,
      row("910729888", "72.9", "nono.txt"),
      row("910733581", "72.9", "nono.txt"),
    ].join("\n");
    const { sermons, skipped } = loadSermons(csv);
    expect(sermons).toHaveLength(1);
    expect(skipped.some((s) => s.reason.includes("duplicate"))).toBe(true);
  });

  it("keeps the higher-scoring duplicate", () => {
    const csv = [header, row("a", "60.0", "nono.txt"), row("b", "80.0", "nono.txt")].join("\n");
    const { sermons } = loadSermons(csv);
    expect(sermons).toHaveLength(1);
    expect(sermons[0]?.score).toBe(80);
    expect(sermons[0]?.id).toBe("b");
  });
});
