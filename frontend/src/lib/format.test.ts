import { describe, expect, it } from "vitest";
import { formatDate, formatDuration, stripLeadingDate } from "./format.ts";

describe("stripLeadingDate", () => {
  it("removes the dashed date prefix the corpus titles carry", () => {
    expect(stripLeadingDate("18-07-2021 - Gênesis 17.9-27")).toBe("Gênesis 17.9-27");
  });

  it("removes a slashed prefix too", () => {
    expect(stripLeadingDate("30/06/2019 - Aula 78 - CFW 28 - Batismo infantil")).toBe(
      "Aula 78 - CFW 28 - Batismo infantil",
    );
  });

  it("leaves titles without a date alone", () => {
    const title = "I Congresso Peregrinos - Santificação";
    expect(stripLeadingDate(title)).toBe(title);
  });

  it("never returns an empty heading", () => {
    expect(stripLeadingDate("18-07-2021 - ")).toBe("18-07-2021 -");
  });
});

describe("formatDate", () => {
  it("renders a short pt-BR date", () => {
    expect(formatDate("2021-07-18")).toBe("18 jul 2021");
  });

  it("passes through anything unparseable", () => {
    expect(formatDate("sem data")).toBe("sem data");
  });
});

describe("formatDuration", () => {
  it("drops a leading zero hour", () => {
    expect(formatDuration("0:45:49")).toBe("45:49");
    expect(formatDuration("1:05:50")).toBe("1:05:50");
  });
});
