import { describe, expect, it } from "vitest";
import { type CountRow, countFacets } from "../src/lib/browse/counts.ts";

/**
 * Facet counts adjusted to the filters already chosen.
 *
 * The point of the whole file: the "+ filtro" popover must never offer a
 * choice that empties the result. Showing the archive-wide total there would
 * do exactly that -- "Gênesis (73)" next to a preacher who never preached it.
 */

const row = (over: Partial<CountRow> = {}): CountRow => ({
  artist: "Reverendo Bruno Melo",
  serviceType: "culto",
  seriesSlug: null,
  date: new Date("2024-03-17T00:00:00Z"),
  scriptures: [{ bookSlug: "efesios", chapter: 5 }],
  topics: [],
  ...over,
});

describe("countFacets", () => {
  it("counts every dimension over an unfiltered corpus", () => {
    const counts = countFacets(
      [
        row(),
        row({ artist: "Pastor Lucas Antunes", serviceType: "ebd", seriesSlug: "cfw-3" }),
        row({ topics: [{ topicSlug: "ansiedade" }] }),
      ],
      {},
    );

    expect(counts.total).toBe(3);
    expect(counts.pregadores).toEqual({ "Reverendo Bruno Melo": 2, "Pastor Lucas Antunes": 1 });
    expect(counts.tipos).toEqual({ culto: 2, ebd: 1 });
    expect(counts.series).toEqual({ "cfw-3": 1 });
    expect(counts.livros).toEqual({ efesios: 3 });
    expect(counts.temas).toEqual({ ansiedade: 1 });
  });

  it("narrows the other dimensions to what the current filter leaves", () => {
    const counts = countFacets(
      [row(), row({ artist: "Pastor Lucas Antunes", serviceType: "ebd" })],
      { pregadores: ["Reverendo Bruno Melo"] },
    );

    // Only the Reverendo's sermon survives, so "ebd" must not be offered.
    expect(counts.tipos).toEqual({ culto: 1 });
    expect(counts.total).toBe(1);
  });

  it("keeps a dimension's own filter out of its own counts", () => {
    // Otherwise picking Bruno Melo would zero every other preacher and nobody
    // could ever add a second one -- the filters are OR within a dimension.
    const counts = countFacets([row(), row({ artist: "Pastor Lucas Antunes" })], {
      pregadores: ["Reverendo Bruno Melo"],
    });

    expect(counts.pregadores).toEqual({ "Reverendo Bruno Melo": 1, "Pastor Lucas Antunes": 1 });
  });

  it("counts a sermon once per book however many chapters it spans", () => {
    // "Gênesis 12-50" is 39 scripture rows and one sermon.
    const counts = countFacets(
      [
        row({
          scriptures: [
            { bookSlug: "genesis", chapter: 12 },
            { bookSlug: "genesis", chapter: 13 },
          ],
        }),
      ],
      {},
    );

    expect(counts.livros).toEqual({ genesis: 1 });
  });

  it("treats the chapter as part of the book filter, not a dimension of its own", () => {
    const rows = [row(), row({ scriptures: [{ bookSlug: "efesios", chapter: 1 }] })];

    // With Efésios 5 chosen, only that sermon counts toward the other facets…
    expect(countFacets(rows, { livros: ["efesios"], capitulo: 5 }).total).toBe(1);
    // …but the book column itself still shows what dropping the chapter opens up.
    expect(countFacets(rows, { livros: ["efesios"], capitulo: 5 }).livros).toEqual({ efesios: 2 });
  });

  it("respects a date range", () => {
    const rows = [row(), row({ date: new Date("2019-01-06T00:00:00Z") })];

    expect(countFacets(rows, { de: "2024-01-01" }).total).toBe(1);
    expect(countFacets(rows, { ate: "2020-01-01" }).total).toBe(1);
    expect(countFacets(rows, { de: "2019-01-01", ate: "2025-01-01" }).total).toBe(2);
  });

  it("offers every year the other filters leave, including the one already chosen", () => {
    // The popover filters by year, so a range that has narrowed to 2024 must
    // still show 2019 -- otherwise switching years is impossible.
    const rows = [row(), row({ date: new Date("2019-01-06T00:00:00Z") })];

    expect(countFacets(rows, { de: "2024-01-01", ate: "2024-12-31" }).anos).toEqual({
      "2024": 1,
      "2019": 1,
    });
  });

  it("ignores a sermon with no type or series rather than inventing a key", () => {
    const counts = countFacets([row({ serviceType: null, seriesSlug: null })], {});

    expect(counts.tipos).toEqual({});
    expect(counts.series).toEqual({});
    expect(counts.total).toBe(1);
  });
});
