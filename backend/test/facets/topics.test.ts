import { describe, expect, it } from "vitest";
import {
  labelRows,
  MAX_TOPICS_PER_SERMON,
  sampleTranscript,
  taxonomyRows,
} from "../../src/lib/facets/topics.ts";

/**
 * The topic taxonomy and the labels against it.
 *
 * Both sides are guarded against the same failure: a model inventing a topic.
 * The taxonomy is committed ground truth, and a label pointing at a topic that
 * is not in it becomes a foreign-key violation at load time -- or worse, a
 * facet page that renders and lists nothing.
 */
const proposal = [
  {
    grupo: "Vida Cristã",
    topico: "Ansiedade e medo",
    descricao: "Sermões sobre ansiedade, medo e a paz de Deus.",
  },
  { grupo: "Vida Cristã", topico: "Perdão", descricao: "Perdoar e ser perdoado." },
  { grupo: "Família e Relacionamentos", topico: "Casamento", descricao: "O casamento cristão." },
];

describe("taxonomyRows", () => {
  it("slugifies both levels and keeps the readable names", () => {
    const [first] = taxonomyRows(proposal);

    expect(first).toEqual({
      grupo_slug: "vida-crista",
      grupo_nome: "Vida Cristã",
      topico_slug: "ansiedade-e-medo",
      topico_nome: "Ansiedade e medo",
      descricao: "Sermões sobre ansiedade, medo e a paz de Deus.",
    });
  });

  it("drops a repeated topic rather than writing a duplicate key", () => {
    const rows = taxonomyRows([...proposal, { ...proposal[1] } as (typeof proposal)[number]]);
    expect(rows).toHaveLength(3);
  });

  it("keeps two topics with the same name under different groups apart", () => {
    // "Sofrimento" under Vida Cristã and under Sofrimento e Esperança are
    // different leaves; collapsing them loses one.
    const rows = taxonomyRows([
      { grupo: "Vida Cristã", topico: "Sofrimento", descricao: "a" },
      { grupo: "Sofrimento e Esperança", topico: "Sofrimento", descricao: "b" },
    ]);

    expect(rows.map((r) => r.topico_slug)).toEqual(["sofrimento", "sofrimento-2"]);
  });

  it("refuses an entry with no name on either level", () => {
    const rows = taxonomyRows([
      { grupo: "", topico: "Perdão", descricao: "x" },
      { grupo: "Vida Cristã", topico: "  ", descricao: "x" },
    ]);

    expect(rows).toEqual([]);
  });
});

describe("sampleTranscript", () => {
  const words = (n: number, prefix = "w") =>
    Array.from({ length: n }, (_, i) => `${prefix}${i}`).join(" ");

  it("returns a sermon shorter than the sample whole, with no elision mark", () => {
    const sample = sampleTranscript(words(20), 10, 10, 10);

    expect(sample).toBe(words(20));
    expect(sample).not.toContain("[…]");
  });

  it("takes the opening, the middle and the close of a long one", () => {
    // The application -- marriage, money, anxiety -- lands in the second half,
    // and a prefix-only sample would label the whole archive by its exegesis.
    const sample = sampleTranscript(words(1000), 10, 10, 10);

    expect(sample).toContain("w0 w1");
    expect(sample).toContain("w495");
    expect(sample).toContain("w999");
    expect(sample).toContain("[…]");
  });

  it("keeps the sample bounded however long the sermon is", () => {
    const sample = sampleTranscript(words(20000), 10, 10, 10);
    expect(sample.split(/\s+/).filter((w) => w !== "[…]")).toHaveLength(30);
  });
});

describe("labelRows", () => {
  const known = new Set(["ansiedade-e-medo", "perdao", "casamento"]);

  it("keeps the labels that exist in the taxonomy", () => {
    const rows = labelRows(known, "s1", [
      { topico_slug: "perdao", confianca: 0.9 },
      { topico_slug: "casamento", confianca: 0.4 },
    ]);

    expect(rows).toEqual([
      { sermon_id: "s1", topico_slug: "perdao", confianca: 0.9 },
      { sermon_id: "s1", topico_slug: "casamento", confianca: 0.4 },
    ]);
  });

  it("silently drops a topic the taxonomy does not have", () => {
    // Otherwise index-facets fails on a foreign key at the end of a paid run.
    expect(labelRows(known, "s1", [{ topico_slug: "escatologia", confianca: 1 }])).toEqual([]);
  });

  it("drops a repeated topic for the same sermon", () => {
    const rows = labelRows(known, "s1", [
      { topico_slug: "perdao", confianca: 0.9 },
      { topico_slug: "perdao", confianca: 0.5 },
    ]);

    expect(rows).toHaveLength(1);
  });

  it("clamps a confidence outside 0..1", () => {
    const rows = labelRows(known, "s1", [
      { topico_slug: "perdao", confianca: 4 },
      { topico_slug: "casamento", confianca: -1 },
    ]);

    expect(rows.map((r) => r.confianca)).toEqual([1, 0]);
  });

  it("caps how many topics a single sermon can carry", () => {
    // A sermon labelled with ten topics is labelled with none: every facet
    // page fills with sermons that only glance at it.
    const many = [...known].concat([...known]).map((s) => ({ topico_slug: s, confianca: 1 }));
    expect(labelRows(known, "s1", many).length).toBeLessThanOrEqual(MAX_TOPICS_PER_SERMON);
  });
});
