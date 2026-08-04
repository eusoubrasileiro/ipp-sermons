import { describe, expect, it } from "vitest";
import { fold, highlight, queryTerms, snippet } from "./highlight.ts";

const marked = (text: string, query: string) =>
  highlight(text, query)
    .filter((p) => p.match)
    .map((p) => p.text);

describe("queryTerms", () => {
  it("drops short words and filler", () => {
    expect(queryTerms("o que a Bíblia diz sobre o batismo")).toEqual(["biblia", "diz", "batismo"]);
  });

  it("de-duplicates", () => {
    expect(queryTerms("graça graça")).toEqual(["graca"]);
  });
});

describe("highlight", () => {
  it("returns the text untouched when there is nothing to mark", () => {
    expect(highlight("texto", "de")).toEqual([{ text: "texto", match: false }]);
  });

  it("matches across accents in both directions", () => {
    // The retrieval side folds accents, so the highlighter has to as well or
    // the passage that caused the hit renders unmarked.
    expect(marked("carta aos Efésios", "efesios")).toEqual(["Efésios"]);
    expect(marked("carta aos Efesios", "Efésios")).toEqual(["Efesios"]);
  });

  it("is case-insensitive and keeps the original casing", () => {
    expect(marked("O Batismo infantil", "batismo")).toEqual(["Batismo"]);
  });

  it("marks word prefixes but not matches inside a word", () => {
    expect(marked("batismos e rebatismo", "batismo")).toEqual(["batismo"]);
  });

  it("preserves the full text across the returned parts", () => {
    const text = "A justificação pela fé é o artigo da fé";
    expect(
      highlight(text, "justificação fé")
        .map((p) => p.text)
        .join(""),
    ).toBe(text);
  });
});

describe("snippet", () => {
  const filler = "palavra ".repeat(120);
  const long = `${filler}o batismo infantil ${filler}`;

  it("leaves short excerpts alone", () => {
    expect(snippet("curto demais", "curto")).toBe("curto demais");
  });

  it("centres the window on the match instead of the opening words", () => {
    const out = snippet(long, "batismo");
    expect(out).toContain("batismo infantil");
    expect(out.length).toBeLessThan(long.length);
    expect(out.startsWith("…")).toBe(true);
    expect(out.endsWith("…")).toBe(true);
  });

  it("falls back to the whole text when nothing matches", () => {
    expect(snippet(long, "zzz")).toBe(long);
  });

  it("cuts on word boundaries", () => {
    const out = snippet(long, "batismo").replace(/^…|…$/g, "");
    expect(out.startsWith("palavra")).toBe(true);
  });
});

describe("fold", () => {
  it("strips diacritics and lowercases", () => {
    expect(fold("Coração Ünico")).toBe("coracao unico");
  });
});
