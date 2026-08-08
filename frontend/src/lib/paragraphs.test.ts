import { describe, expect, it } from "vitest";
import { CHUNK_STRIDE, paragraphAtWord } from "./paragraphs.ts";

/**
 * Anchoring is the browser's business alone: it exists to scroll the reader to
 * the passage that answered their search. The splitting it counts against
 * lives in `@ipp/shared`, tested there, because the server renders the same
 * paragraphs into the page a crawler reads.
 */

const sentence = (words: number, word = "palavra") => `${Array(words).fill(word).join(" ")}.`;

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
