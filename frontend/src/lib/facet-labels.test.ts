import { describe, expect, it } from "vitest";
import { FACETS } from "../facet-fixtures.ts";
import { chipsOf, optionsOf } from "./facet-labels.ts";

/**
 * What a filter is called on screen.
 *
 * A filter travels as a slug, a preacher's full name or a date range; none of
 * those may reach the chips, and a chip that shows "efesios" tells a visitor
 * the site is broken.
 */
const facets = FACETS as unknown as Parameters<typeof optionsOf>[1];

describe("chipsOf", () => {
  it("names every dimension in Portuguese", () => {
    const chips = chipsOf(
      {
        livros: ["efesios"],
        series: ["cfw-3"],
        pregadores: ["Reverendo Bruno Melo"],
        tipos: ["culto"],
        temas: ["ansiedade"],
      },
      facets,
    );

    expect(chips.map((c) => `${c.dimensao}: ${c.label}`)).toEqual([
      "Pregador: Bruno Melo",
      "Tipo: Culto",
      "Série: CFW 3 — Decreto",
      "Bíblia: Efésios",
      "Tema: Ansiedade",
    ]);
  });

  it("falls back to the raw value before the index tree has loaded", () => {
    // The chips render on the first paint of a shared link; a blank chip then
    // would be worse than an unlovely one.
    expect(chipsOf({ livros: ["efesios"] }, null)[0]?.label).toBe("efesios");
  });

  it("leaves a value the tree does not know as it is", () => {
    expect(chipsOf({ livros: ["inventado"] }, facets)[0]?.label).toBe("inventado");
  });

  it("shows a whole year as the year", () => {
    expect(chipsOf({ de: "2024-01-01", ate: "2024-12-31" }, facets)[0]).toMatchObject({
      dimensao: "Período",
      label: "2024",
    });
  });

  it("spells out a range that is not a year", () => {
    // The picker only ever sets whole years, but a shared link can carry
    // anything, and calling 2024-2025 "2024" would be a lie.
    expect(chipsOf({ de: "2024-06-01", ate: "2025-06-01" }, facets)[0]?.label).toBe(
      "de 2024-06-01 até 2025-06-01",
    );
    expect(chipsOf({ ate: "2020-01-01" }, facets)[0]?.label).toBe("até 2020-01-01");
  });
});

describe("optionsOf", () => {
  it("filters preachers by the full name, not the slug", () => {
    // hybrid_search() compares against sermons.artist.
    expect(optionsOf("pregadores", facets, null)[0]).toMatchObject({
      value: "Reverendo Bruno Melo",
      label: "Bruno Melo",
    });
  });

  it("keeps the canonical order of the index pages", () => {
    expect(optionsOf("livros", facets, null).map((o) => o.label)).toEqual(["Gênesis", "Efésios"]);
  });

  it("reads zero for a facet the counts do not mention", () => {
    expect(optionsOf("temas", facets, null)[0]?.total).toBe(0);
  });
});
