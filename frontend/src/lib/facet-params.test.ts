import { describe, expect, it } from "vitest";
import {
  addFilter,
  countFilters,
  dropFilter,
  dropYear,
  parseFilters,
  toSearchParams,
  withYear,
} from "./facet-params.ts";

/**
 * The filter state lives in the URL, so this is the contract behind every
 * shareable link: /?q=briga&livros=efesios&capitulo=5 has to survive a reload,
 * a WhatsApp forward and a back button.
 */

const sp = (s: string) => new URLSearchParams(s);

describe("parseFilters", () => {
  it("reads a comma-separated list", () => {
    expect(parseFilters(sp("livros=efesios,genesis"))).toEqual({ livros: ["efesios", "genesis"] });
  });

  it("keeps an absent facet absent rather than empty", () => {
    // An empty array means "match nothing" all the way down to the SQL, so an
    // unfiltered search must not send one.
    expect(parseFilters(sp("q=lei"))).toEqual({});
    expect(parseFilters(sp("livros="))).toEqual({});
  });

  it("reads the chapter as a number and ignores nonsense", () => {
    expect(parseFilters(sp("livros=efesios&capitulo=5")).capitulo).toBe(5);
    expect(parseFilters(sp("livros=efesios&capitulo=abc")).capitulo).toBeUndefined();
  });

  it("drops a chapter with no book to narrow", () => {
    expect(parseFilters(sp("capitulo=5"))).toEqual({});
  });

  it("keeps only well-formed dates", () => {
    expect(parseFilters(sp("de=2024-01-01&ate=2024-12-31"))).toEqual({
      de: "2024-01-01",
      ate: "2024-12-31",
    });
    expect(parseFilters(sp("de=ontem"))).toEqual({});
  });
});

describe("toSearchParams", () => {
  it("round-trips through the address bar", () => {
    const filters = { livros: ["efesios"], capitulo: 5, pregadores: ["Reverendo Bruno Melo"] };
    const round = parseFilters(toSearchParams("briga na igreja", filters));

    expect(round).toEqual(filters);
  });

  it("carries the query and omits it when there is none", () => {
    expect(toSearchParams("lei moral", {}).get("q")).toBe("lei moral");
    expect(toSearchParams("", { tipos: ["ebd"] }).has("q")).toBe(false);
  });

  it("writes nothing for a facet with no values", () => {
    expect(toSearchParams("lei", { livros: [] }).toString()).toBe("q=lei");
  });
});

describe("addFilter / dropFilter", () => {
  it("adds without mutating what it was given", () => {
    const before = { livros: ["efesios"] };
    expect(addFilter(before, "livros", "genesis")).toEqual({ livros: ["efesios", "genesis"] });
    expect(before).toEqual({ livros: ["efesios"] });
  });

  it("never adds the same value twice", () => {
    expect(addFilter({ tipos: ["ebd"] }, "tipos", "ebd")).toEqual({ tipos: ["ebd"] });
  });

  it("removes the facet entirely when its last value goes", () => {
    expect(dropFilter({ tipos: ["ebd"], livros: ["efesios"] }, "tipos", "ebd")).toEqual({
      livros: ["efesios"],
    });
  });

  it("takes the chapter with the last book", () => {
    // A chapter with no book filters nothing and would read as a stray chip.
    expect(dropFilter({ livros: ["efesios"], capitulo: 5 }, "livros", "efesios")).toEqual({});
  });
});

describe("withYear", () => {
  it("turns a year into the range the API takes", () => {
    expect(withYear({}, 2024)).toEqual({ de: "2024-01-01", ate: "2024-12-31" });
  });

  it("replaces the year rather than intersecting with the old one", () => {
    expect(withYear({ de: "2019-01-01", ate: "2019-12-31" }, 2024).de).toBe("2024-01-01");
  });

  it("drops the range again", () => {
    expect(dropYear({ de: "2024-01-01", ate: "2024-12-31", tipos: ["ebd"] })).toEqual({
      tipos: ["ebd"],
    });
  });
});

describe("countFilters", () => {
  it("counts one per chip, with the date range as a single chip", () => {
    expect(countFilters({})).toBe(0);
    expect(countFilters({ livros: ["efesios"], capitulo: 5 })).toBe(1);
    expect(countFilters({ livros: ["efesios", "genesis"], tipos: ["ebd"] })).toBe(3);
    expect(countFilters({ de: "2024-01-01", ate: "2024-12-31" })).toBe(1);
  });
});
