import { describe, expect, it } from "vitest";
import { parseCsv } from "../../src/lib/corpus.ts";
import { writeCsv } from "../../src/lib/facets/csv.ts";

describe("writeCsv", () => {
  it("writes a header and the rows in column order", () => {
    const csv = writeCsv(["b", "a"], [{ a: "1", b: "2" }]);
    expect(csv).toBe("b,a\n2,1\n");
  });

  it("writes a header-only file when there are no rows", () => {
    expect(writeCsv(["a", "b"], [])).toBe("a,b\n");
  });

  it("leaves plain values unquoted, so a diff stays small", () => {
    expect(writeCsv(["name"], [{ name: "O Livro dos Reis" }])).toBe("name\nO Livro dos Reis\n");
  });

  it("quotes only what would otherwise break the format", () => {
    // Sermon titles contain commas and quotation marks routinely.
    const csv = writeCsv(
      ["title"],
      [{ title: "Ser como Paulo, o pior" }, { title: 'diz "paz" a todos' }],
    );
    expect(csv).toBe('title\n"Ser como Paulo, o pior"\n"diz ""paz"" a todos"\n');
  });

  it("renders a missing or null cell as empty", () => {
    expect(writeCsv(["a", "b", "c"], [{ a: null, b: undefined }])).toBe("a,b,c\n,,\n");
  });

  it("renders numbers and booleans without quoting", () => {
    expect(writeCsv(["n", "f"], [{ n: 27, f: true }])).toBe("n,f\n27,true\n");
  });

  it("round-trips through parseCsv", () => {
    const rows = [
      { id: "1820303268", title: "Ser como Paulo, o pior", part: 2 },
      { id: "x", title: 'diz "paz"\nem duas linhas', part: null },
    ];
    const back = parseCsv(writeCsv(["id", "title", "part"], rows));
    expect(back).toEqual([
      { id: "1820303268", title: "Ser como Paulo, o pior", part: "2" },
      { id: "x", title: 'diz "paz"\nem duas linhas', part: "" },
    ]);
  });
});
