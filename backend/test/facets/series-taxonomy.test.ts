import { describe, expect, it } from "vitest";
import type { NameCluster } from "../../src/lib/facets/cluster.ts";
import {
  buildSeriesRows,
  kindOf,
  lenientKey,
  retiredSlugs,
  SERIES_COLUMNS,
  type SeriesDecision,
  type SeriesRow,
} from "../../src/lib/facets/series-taxonomy.ts";

const cluster = (provisional: string, count: number, members?: string[]): NameCluster => ({
  provisional,
  members: members ?? [provisional],
  count,
});

const decide = (id: number, name: string, extra: Partial<SeriesDecision> = {}): SeriesDecision => ({
  id,
  name,
  description: "",
  parent: null,
  merge_into: null,
  ...extra,
});

describe("lenientKey", () => {
  it("drops a leading article", () => {
    expect(lenientKey("A Confissão de Fé de Westminster")).toBe("confissao-de-fe-de-westminster");
    expect(lenientKey("Confissão de Fé de Westminster")).toBe("confissao-de-fe-de-westminster");
  });

  it("keeps an article that is part of the name", () => {
    expect(lenientKey("O Livro dos Reis")).toBe("livro-dos-reis");
    expect(lenientKey("Livro dos Reis")).toBe("livro-dos-reis");
  });
});

describe("kindOf", () => {
  it("recognises the Westminster chapters", () => {
    expect(kindOf("CFW 3 — Do Decreto Eterno de Deus")).toBe("cfw");
    expect(kindOf("CFW 23")).toBe("cfw");
  });

  it("recognises the events", () => {
    expect(kindOf("IV Conferência Peregrinos")).toBe("conferencia");
    expect(kindOf("I Congresso Peregrinos")).toBe("congresso");
    expect(kindOf("Confraria Peregrinos")).toBe("confraria");
    expect(kindOf("Diaconia")).toBe("diaconia");
  });

  it("defaults to Sunday school", () => {
    expect(kindOf("O Livro dos Reis")).toBe("ebd");
  });
});

describe("buildSeriesRows", () => {
  it("takes the adjudicated name over the raw one", () => {
    const rows = buildSeriesRows(
      [cluster("Atribututos de Deus", 11, ["Atribututos de Deus", "Atributos de Deus"])],
      [decide(0, "Atributos de Deus")],
    );
    expect(rows[0]).toMatchObject({
      slug: "atributos-de-deus",
      name: "Atributos de Deus",
      sermon_count: 11,
      variants: "Atribututos de Deus|Atributos de Deus",
    });
  });

  it("falls back to the raw name when the model returns none", () => {
    const rows = buildSeriesRows([cluster("Apologética", 1)], [decide(0, "   ")]);
    expect(rows[0]?.name).toBe("Apologética");
  });

  it("links a parent whose name gained a definite article", () => {
    // The bug this exists to prevent: one article orphaned all twelve chapters.
    const rows = buildSeriesRows(
      [cluster("A Confissão de Fé de Westminster", 1), cluster("CFW 3", 5)],
      [
        decide(0, "A Confissão de Fé de Westminster"),
        decide(1, "CFW 3 — Do Decreto Eterno de Deus", {
          parent: "Confissão de Fé de Westminster",
        }),
      ],
    );

    const slugs = new Set(rows.map((r) => r.slug));
    const parents = rows.map((r) => r.parent_slug).filter(Boolean) as string[];
    expect(parents).toHaveLength(1);
    expect(slugs.has(parents[0] as string)).toBe(true);
  });

  it("types a row that other rows point at as the head of the course", () => {
    const rows = buildSeriesRows(
      [cluster("Confissão de Fé de Westminster", 1), cluster("CFW 3", 5)],
      [
        decide(0, "Confissão de Fé de Westminster"),
        decide(1, "CFW 3", { parent: "Confissão de Fé de Westminster" }),
      ],
    );
    expect(rows.find((r) => r.slug === "confissao-de-fe-de-westminster")?.kind).toBe("cfw");
  });

  it("keeps a parent label that resolves to no row", () => {
    const rows = buildSeriesRows(
      [cluster("CFW 3", 5)],
      [decide(0, "CFW 3", { parent: "Confissão de Fé de Westminster" })],
    );
    expect(rows[0]?.parent_slug).toBe("confissao-de-fe-de-westminster");
    expect(rows[0]?.parent_name).toBe("Confissão de Fé de Westminster");
  });

  it("merges two clusters the model declared the same course", () => {
    const rows = buildSeriesRows(
      [cluster("Sola Fide", 2), cluster("Sola Fide: Somente a Fé", 1)],
      [decide(0, "Sola Fide"), decide(1, "Sola Fide", { merge_into: 0 })],
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]?.sermon_count).toBe(3);
    expect(rows[0]?.variants).toBe("Sola Fide|Sola Fide: Somente a Fé");
  });

  it("survives a merge cycle rather than recursing forever", () => {
    const rows = buildSeriesRows(
      [cluster("A", 1), cluster("B", 1)],
      [decide(0, "A", { merge_into: 1 }), decide(1, "B", { merge_into: 0 })],
    );
    expect(rows.length).toBeGreaterThanOrEqual(1);
  });

  it("orders by size so the index leads with the real courses", () => {
    const rows = buildSeriesRows(
      [cluster("Apologética", 1), cluster("O Livro dos Reis", 17), cluster("Diaconia", 7)],
      [decide(0, "Apologética"), decide(1, "O Livro dos Reis"), decide(2, "Diaconia")],
    );
    expect(rows.map((r) => r.sermon_count)).toEqual([17, 7, 1]);
  });

  it("returns nothing for no clusters", () => {
    const rows: SeriesRow[] = buildSeriesRows([], []);
    expect(rows).toEqual([]);
  });

  it("fills every column the CSV writer asks for", () => {
    // The row shape and SERIES_COLUMNS have to agree, or a committed column
    // silently writes empty for every series.
    const [row] = buildSeriesRows([cluster("Diaconia", 7)], [decide(0, "Diaconia")]);
    for (const column of SERIES_COLUMNS) {
      expect(row).toHaveProperty(column);
    }
  });
});

describe("retiredSlugs", () => {
  /**
   * canonicalize:series rewrites series.csv in full from a non-deterministic
   * model answer. Adding a series is harmless; losing one is not -- a slug that
   * is already committed is a /series URL somebody may have linked to, and the
   * merge that removes it is exactly what the pass is designed to do.
   *
   * The line is git: a slug in the last commit is live, a slug that only exists
   * in this run is not.
   */
  it("says nothing when the taxonomy only grows", () => {
    expect(
      retiredSlugs(["cfw-1", "o-livro-dos-reis"], ["cfw-1", "o-livro-dos-reis", "cfw-2"]),
    ).toEqual([]);
  });

  it("catches a committed slug the model merged away", () => {
    expect(retiredSlugs(["cfw-1", "atributos-de-deus"], ["cfw-1", "os-atributos-de-deus"])).toEqual(
      ["atributos-de-deus"],
    );
  });

  it("catches every one of them, not just the first", () => {
    expect(retiredSlugs(["a", "b", "c"], ["c"])).toEqual(["a", "b"]);
  });

  it("treats an absent baseline as nothing to protect", () => {
    expect(retiredSlugs([], ["cfw-1"])).toEqual([]);
  });
});

describe("buildSeriesRows with a committed taxonomy", () => {
  /**
   * The slug is a URL and the name is a label, and deriving one from the other
   * means every re-wording of a series title silently moves its page.
   *
   * The model re-reads the Westminster chapter names from example titles on
   * every run, so "Da Escritura Sagrada" and "Da Sagrada Escritura" are both
   * reachable answers for the same chapter. Once published, the slug has to stop
   * following the name -- `variants` is what says the two are the same series.
   */
  const committed = new Map([
    ["cfw-1", "cfw-1-da-escritura-sagrada"],
    ["cfw-1-da-escritura-sagrada", "cfw-1-da-escritura-sagrada"],
  ]);

  it("keeps the published slug when the model re-words the name", () => {
    const [row] = buildSeriesRows(
      [cluster("CFW 1", 4)],
      [decide(0, "CFW 1 — Da Sagrada Escritura")],
      committed,
    );

    expect(row).toMatchObject({
      slug: "cfw-1-da-escritura-sagrada",
      name: "CFW 1 — Da Sagrada Escritura",
    });
  });

  it("still slugifies a series nobody has published yet", () => {
    const [row] = buildSeriesRows(
      [cluster("CFW 9", 3)],
      [decide(0, "CFW 9 — Do Livre-Arbítrio")],
      committed,
    );

    expect(row?.slug).toBe("cfw-9-do-livre-arbitrio");
  });

  it("behaves as before when there is no committed taxonomy", () => {
    const [row] = buildSeriesRows(
      [cluster("CFW 1", 4)],
      [decide(0, "CFW 1 — Da Sagrada Escritura")],
    );

    expect(row?.slug).toBe("cfw-1-da-sagrada-escritura");
  });

  it("folds two clusters onto one published slug when the model merges them", () => {
    const rows = buildSeriesRows(
      [cluster("CFW 1", 4), cluster("CFW 1 - Escritura", 2)],
      [decide(0, "CFW 1 — Da Escritura"), decide(1, "CFW 1 — Da Escritura", { merge_into: 0 })],
      new Map([...committed, ["cfw-1-escritura", "cfw-1-da-escritura-sagrada"]]),
    );

    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ slug: "cfw-1-da-escritura-sagrada", sermon_count: 6 });
  });
});
