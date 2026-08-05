import { describe, expect, it } from "vitest";
import { buildVariantIndex, resolveSeries } from "../../src/lib/facets/variants.ts";

const row = (slug: string, variants: string) => ({ slug, variants });

describe("buildVariantIndex", () => {
  it("maps a renamed series back from the name the title used", () => {
    // The real pair: the title says "CFW 3", the taxonomy says something longer.
    const index = buildVariantIndex([row("cfw-3-do-decreto-eterno-de-deus", "CFW 3")]);
    expect(resolveSeries(index, "cfw-3")).toBe("cfw-3-do-decreto-eterno-de-deus");
  });

  it("maps a series to itself", () => {
    const index = buildVariantIndex([row("diaconia", "Diaconia")]);
    expect(resolveSeries(index, "diaconia")).toBe("diaconia");
  });

  it("maps every spelling a cluster absorbed", () => {
    const index = buildVariantIndex([
      row("atributos-de-deus", "Atribututos de Deus|Atributos de Deus"),
    ]);
    expect(resolveSeries(index, "atribututos-de-deus")).toBe("atributos-de-deus");
    expect(resolveSeries(index, "atributos-de-deus")).toBe("atributos-de-deus");
  });

  it("returns null for a series nothing claims", () => {
    const index = buildVariantIndex([row("diaconia", "Diaconia")]);
    expect(resolveSeries(index, "serie-nova")).toBeNull();
  });

  it("returns null for an empty slug", () => {
    expect(resolveSeries(buildVariantIndex([]), "  ")).toBeNull();
  });

  it("ignores a row with no slug", () => {
    expect(buildVariantIndex([row("", "Alguma coisa")]).size).toBe(0);
  });

  it("ignores empty entries in the variant list", () => {
    const index = buildVariantIndex([row("diaconia", "Diaconia||  |")]);
    expect(index.size).toBe(1);
  });
});
