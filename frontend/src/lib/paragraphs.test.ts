import { describe, expect, it } from "vitest";
import { CHUNK_STRIDE, paragraphAtWord, toParagraphs } from "./paragraphs.ts";

/**
 * The corpus arrives as one unbroken block — zero newline characters across all
 * 503 committed transcripts — so every paragraph the reader sees is invented
 * here. These tests pin the two things that must hold for that to be safe: no
 * word is ever lost, and the anchor still lands on the passage that matched.
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

describe("paragraphAtWord", () => {
  const paragraphs = [sentence(100), sentence(100), sentence(100)];

  it("finds the paragraph holding a word offset", () => {
    expect(paragraphAtWord(paragraphs, 0)).toBe(0);
    expect(paragraphAtWord(paragraphs, 50)).toBe(0);
    expect(paragraphAtWord(paragraphs, 150)).toBe(1);
    expect(paragraphAtWord(paragraphs, 250)).toBe(2);
  });

  it("maps a chunk index through the chunker's stride", () => {
    // chunkText(text, 200, 30) steps 170 words, so chunk 2 opens at word 340 --
    // inside the fourth 100-word paragraph.
    const longer = Array.from({ length: 6 }, () => sentence(100));
    expect(paragraphAtWord(longer, 2 * CHUNK_STRIDE)).toBe(3);
  });

  it("falls back to the top rather than off the end", () => {
    // A link from an older index can name a chunk the sermon no longer has.
    expect(paragraphAtWord(paragraphs, 99_999)).toBe(0);
    expect(paragraphAtWord([], 10)).toBe(0);
  });
});
