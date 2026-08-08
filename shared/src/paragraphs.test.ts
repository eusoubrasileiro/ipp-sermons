import { describe, expect, it } from "vitest";
import { countWords, toParagraphs } from "./paragraphs.ts";

/**
 * The corpus arrives as one unbroken block — zero newline characters across all
 * 503 committed transcripts — so every paragraph a reader sees is invented
 * here. These tests pin what must hold for that to be safe: no word is ever
 * lost, and a scripture reference is never cut in half.
 *
 * They live beside the splitter rather than in `frontend/` because the server
 * renders these same paragraphs into the page a crawler reads.
 */

const sentence = (words: number, word = "palavra") => `${Array(words).fill(word).join(" ")}.`;

function allWords(paragraphs: string[]): string[] {
  return paragraphs.join(" ").split(/\s+/).filter(Boolean);
}

describe("toParagraphs", () => {
  it("splits a wall of text on sentence boundaries", () => {
    const got = toParagraphs("Primeira frase. Segunda frase! Terceira frase?", 2);
    expect(got).toEqual(["Primeira frase.", "Segunda frase!", "Terceira frase?"]);
  });

  it("groups sentences up to the target length", () => {
    const text = [sentence(30), sentence(30), sentence(30), sentence(30)].join(" ");
    const got = toParagraphs(text, 100);

    // 30+30+30 reaches 90, the fourth crosses 100 and closes the paragraph.
    expect(got).toHaveLength(1);
    expect(allWords(got)).toHaveLength(120);
  });

  it("never drops the tail", () => {
    // The last sentences of a sermon are shorter than the target, so a
    // close-only-on-target loop would silently stop rendering the ending.
    const text = [sentence(120), sentence(5)].join(" ");
    const got = toParagraphs(text, 100);

    expect(got).toHaveLength(2);
    expect(got[1]).toBe(sentence(5));
  });

  it("keeps every word of the transcript", () => {
    const text = Array.from({ length: 40 }, (_, i) => sentence(18, `w${i}`)).join(" ");
    expect(allWords(toParagraphs(text))).toEqual(text.split(/\s+/).filter(Boolean));
  });

  it("does not break on a decimal or a scripture reference", () => {
    // "Eclesiastes 3.2b-3a" and "1.5-7" are everywhere in this corpus. A naive
    // split on "." would cut them in half.
    const got = toParagraphs("Lemos Eclesiastes 3.2 hoje. E também 2 Pedro 1.5-7 depois.", 1000);
    expect(got).toEqual(["Lemos Eclesiastes 3.2 hoje. E também 2 Pedro 1.5-7 depois."]);
  });

  it("breaks before a sentence that does not start with a capital", () => {
    // Portuguese sermons open sentences with "e", "mas" and numerals
    // constantly; requiring a capital after the period would join them all.
    expect(toParagraphs("Ele falou. e o povo ouviu.", 1)).toEqual([
      "Ele falou.",
      "e o povo ouviu.",
    ]);
  });

  it("returns nothing for an empty transcript", () => {
    expect(toParagraphs("")).toEqual([]);
    expect(toParagraphs("   \n  ")).toEqual([]);
  });

  it("handles a transcript that ends mid-sentence", () => {
    // Exactly what a truncated download used to produce, and what the reading
    // page must still render rather than blank.
    expect(toParagraphs("nós vamos ficar olhando argumentos para cá e para ")).toEqual([
      "nós vamos ficar olhando argumentos para cá e para",
    ]);
  });
});

describe("countWords", () => {
  it("counts across any run of whitespace", () => {
    expect(countWords("uma   duas\ntrês ")).toBe(3);
  });

  it("counts nothing in an empty string", () => {
    expect(countWords("   ")).toBe(0);
  });
});
