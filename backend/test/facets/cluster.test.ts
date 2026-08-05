import { describe, expect, it } from "vitest";
import {
  clusterNames,
  digitSignature,
  type NameEntry,
  similarity,
} from "../../src/lib/facets/cluster.ts";

describe("similarity", () => {
  it("scores an identical pair as 1", () => {
    expect(similarity("Diaconia", "Diaconia")).toBe(1);
  });

  it("scores a one-letter typo close to 1", () => {
    // The real pair in the corpus: eight lessons spelled one way, three the other.
    expect(similarity("Atribututos de Deus", "Atributos de Deus")).toBeGreaterThan(0.88);
  });

  it("ignores case and accents", () => {
    expect(similarity("ADORAÇÃO BÍBLICA", "adoracao biblica")).toBe(1);
  });

  it("ignores word order", () => {
    expect(similarity("Bispos e Pastores", "Pastores e Bispos")).toBeGreaterThan(0.88);
  });

  it("keeps genuinely different courses apart", () => {
    expect(similarity("Escolhendo Presbíteros", "Escolhendo Diáconos")).toBeLessThan(0.88);
    expect(similarity("O Livro dos Reis", "O dia do Senhor")).toBeLessThan(0.88);
  });

  it("scores two empty strings as 1 rather than dividing by zero", () => {
    expect(similarity("", "")).toBe(1);
    expect(similarity("Diaconia", "")).toBe(0);
  });
});

describe("digitSignature", () => {
  it("collects the numbers in order", () => {
    expect(digitSignature("CFW 23")).toBe("23");
    expect(digitSignature("Aula 4 - CFW 3")).toBe("4,3");
  });

  it("is empty when there are no numbers", () => {
    expect(digitSignature("O Livro dos Reis")).toBe("");
  });

  it("counts roman numerals too", () => {
    // The annual conferences are numbered in roman; an arabic-only signature
    // merged all five editions into one series.
    expect(digitSignature("IV Conferência Peregrinos")).toBe("4");
    expect(digitSignature("V Conferência Peregrinos")).toBe("5");
    expect(digitSignature("I Congresso Peregrinos")).toBe("1");
  });
});

describe("clusterNames", () => {
  const entry = (name: string, count: number): NameEntry => ({ name, count });

  it("merges a typo into its correct spelling", () => {
    const clusters = clusterNames([entry("Atribututos de Deus", 8), entry("Atributos de Deus", 3)]);
    expect(clusters).toHaveLength(1);
    expect(clusters[0]?.members).toHaveLength(2);
    expect(clusters[0]?.count).toBe(11);
  });

  it("never merges names whose numbers differ", () => {
    // CFW 1..28 are twenty-eight different chapters of the Confession and are
    // textually almost identical. Merging them would collapse the whole course
    // into one series.
    const clusters = clusterNames([
      entry("CFW 2", 4),
      entry("CFW 3", 5),
      entry("CFW 23", 4),
      entry("CFW 28", 1),
    ]);
    expect(clusters).toHaveLength(4);
  });

  it("never merges the annual conferences with each other", () => {
    const clusters = clusterNames([
      entry("II Conferência Peregrinos", 3),
      entry("III Conferência Peregrinos", 3),
      entry("IV Conferência Peregrinos", 4),
      entry("V Conferência Peregrinos", 4),
      entry("VI Conferência Peregrinos", 4),
    ]);
    expect(clusters).toHaveLength(5);
  });

  it("still merges names that share the same numbers", () => {
    const clusters = clusterNames([entry("CFW 3", 5), entry("CFW  3", 1)]);
    expect(clusters).toHaveLength(1);
  });

  it("keeps distinct courses distinct", () => {
    const clusters = clusterNames([
      entry("Escolhendo Presbíteros", 4),
      entry("Escolhendo Diáconos", 3),
      entry("O Livro dos Reis", 17),
    ]);
    expect(clusters).toHaveLength(3);
  });

  it("proposes the most-used spelling as the provisional name", () => {
    // Provisional only: here it picks the typo, which is exactly what the LLM
    // adjudication stage exists to correct.
    const clusters = clusterNames([entry("Atributos de Deus", 3), entry("Atribututos de Deus", 8)]);
    expect(clusters[0]?.provisional).toBe("Atribututos de Deus");
  });

  it("groups transitively", () => {
    // a~b and b~c must land in one cluster even when a and c alone fall short.
    const clusters = clusterNames([
      entry("Adoração Bíblica", 2),
      entry("Adoracao Biblica", 1),
      entry("Adoração Biblica", 1),
    ]);
    expect(clusters).toHaveLength(1);
    expect(clusters[0]?.count).toBe(4);
  });

  it("orders clusters by size, so the index leads with the real courses", () => {
    const clusters = clusterNames([
      entry("Apologética", 1),
      entry("O Livro dos Reis", 17),
      entry("Diaconia", 7),
    ]);
    expect(clusters.map((c) => c.count)).toEqual([17, 7, 1]);
  });

  it("returns nothing for no input", () => {
    expect(clusterNames([])).toEqual([]);
  });
});
