import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { type BibleBook, loadBibleBooks } from "../../src/lib/facets/bible.ts";
import {
  type ParsedTitle,
  parseTitle,
  type ServiceType,
} from "../../src/lib/facets/parse-title.ts";

/** Every value the service-type facet is allowed to take. */
const SERVICE_TYPES: ServiceType[] = [
  "culto",
  "ebd",
  "conferencia",
  "congresso",
  "confraria",
  "diaconia",
];

const CSV_PATH = join(import.meta.dirname, "../../../data/facets/bible_books.csv");

describe("parseTitle", () => {
  let books: BibleBook[];

  beforeAll(async () => {
    books = loadBibleBooks(await readFile(CSV_PATH, "utf8"));
  });

  const parse = (title: string, description?: string): ParsedTitle =>
    parseTitle(books, title, description);

  it("always returns a known service type, whatever the title looks like", () => {
    // The union is what the database column and the UI filter agree on; a
    // parser that invented a seventh value would break both silently.
    const oddities = [
      "",
      "   ",
      "O Primeiro Mandamento",
      "I João 3.19-24",
      "22-12-2019  - O Sexto Mandamento (1)",
      "IV Conferência Peregrinos_25-09-2022 - Santificação (pt.2)",
    ];
    for (const title of oddities) {
      expect(SERVICE_TYPES).toContain(parse(title).serviceType);
    }
  });

  it("returns every field even for an empty title", () => {
    expect(parse("")).toEqual({
      serviceType: "culto",
      eventName: null,
      seriesCandidate: null,
      part: null,
      displayTitle: "",
      scripture: null,
    });
  });

  describe("service type", () => {
    it("defaults to the Sunday service", () => {
      expect(parse("28-04-2024 - Efésios 6.1-3 - Filhos").serviceType).toBe("culto");
    });

    it("recognises Sunday school and its misspellings", () => {
      // The corpus writes EBD, ED and EDB for the same thing.
      expect(parse("28-04-2024 - EBD - Atos 27 - 28").serviceType).toBe("ebd");
      expect(parse("11-10-2020 - ED - O nono mandamento").serviceType).toBe("ebd");
      expect(parse("02-01-2022 - EDB - Diaconia Aula 1").serviceType).toBe("ebd");
    });

    it("recognises the conferences and congresses", () => {
      expect(parse("IV Conferência Peregrinos_24-09-2022 - Cristologia").serviceType).toBe(
        "conferencia",
      );
      expect(parse("I Congresso Peregrinos - Santificação - Perigos").serviceType).toBe(
        "congresso",
      );
      expect(parse("14-03-2020 - Confraria Peregrinos").serviceType).toBe("confraria");
    });

    it("reads the conference off the description when the title omits it", () => {
      // 17-09-2021 carries the event only in its SoundCloud description.
      const parsed = parse(
        "17-09-2021 - A Lei Moral e a Vida Cristã - Piedade e Não Legalismo (1)",
        "III Conferência Peregrinos - A Lei Moral e a Vida Cristã por Pastor Alan Kleber",
      );
      expect(parsed.serviceType).toBe("conferencia");
      expect(parsed.eventName).toBe("III Conferência Peregrinos");
    });

    it("treats the standalone deacon course as its own type", () => {
      expect(parse("02-02-2020 - Diaconia - Aula 1").serviceType).toBe("diaconia");
      // ...but when it is taught in Sunday school, Sunday school wins.
      expect(parse("09-01-2022 - EBD - Diaconia Aula 2").serviceType).toBe("ebd");
    });
  });

  describe("series", () => {
    it("takes the segment after EBD as the series", () => {
      expect(parse("25-06-2023 - EBD - Escolhendo Presbíteros - Aula 2").seriesCandidate).toBe(
        "Escolhendo Presbíteros",
      );
      expect(
        parse("05-09-2021 - EBD - Presbíteros, Bispos e Pastores - Aula 3").seriesCandidate,
      ).toBe("Presbíteros, Bispos e Pastores");
    });

    it("strips a run-on lesson marker from the series name", () => {
      expect(parse("09-01-2022 - EBD - Diaconia Aula 2").seriesCandidate).toBe("Diaconia");
    });

    it("does not split one course in two over a lesson marker", () => {
      // These four titles are one course; the markers must not reach the name.
      expect(parse("06-11-2022 - EBD - Igreja e Comunhão").seriesCandidate).toBe(
        "Igreja e Comunhão",
      );
      expect(parse("13-11-2022 - EBD - Igreja e Comunhão (2)").seriesCandidate).toBe(
        "Igreja e Comunhão",
      );
      expect(parse("21-05-2023 - EBD - Permanência no casamento I").seriesCandidate).toBe(
        "Permanência no casamento",
      );
      expect(parse("04-06-2023 - EBD - Permanência no casamento II").seriesCandidate).toBe(
        "Permanência no casamento",
      );
    });

    it("does not invent a series from a bare scripture reference", () => {
      // "EBD - Atos 27 - 28" is a passage, not a course name.
      expect(parse("28-04-2024 - EBD - Atos 27 - 28").seriesCandidate).toBeNull();
      expect(parse("20-12-2020 - EBD - Números 15-36").seriesCandidate).toBeNull();
    });

    it("keeps a course name that merely contains a book word", () => {
      // "Reis" alone is not a book; "O Livro dos Reis" is a 17-part course.
      expect(parse("12-02-2023 - EBD - O Livro dos Reis - 1 Reis 3-4").seriesCandidate).toBe(
        "O Livro dos Reis",
      );
    });

    it("groups the Westminster Confession by its chapter", () => {
      expect(parse("23-06-2024 - EBD - CFW 3 - Do Eterno Decreto de Deus Parte I")).toMatchObject({
        seriesCandidate: "CFW 3",
        part: 1,
      });
      expect(parse("30/06/2019 - Aula 78 - CFW 28 - Batismo infantil").seriesCandidate).toBe(
        "CFW 28",
      );
    });

    it("uses the event as the series for conferences", () => {
      expect(parse("IV Conferência Peregrinos_24-09-2022 - Cristologia").seriesCandidate).toBe(
        "IV Conferência Peregrinos",
      );
      expect(parse("I Congresso Peregrinos - Santificação - Perigos").seriesCandidate).toBe(
        "I Congresso Peregrinos",
      );
    });

    it("leaves an ordinary Sunday sermon without a series", () => {
      expect(parse("28-04-2024 - Efésios 6.1-3 - Filhos").seriesCandidate).toBeNull();
    });
  });

  describe("part", () => {
    it("reads a numbered lesson", () => {
      expect(parse("25-06-2023 - EBD - Escolhendo Presbíteros - Aula 2").part).toBe(2);
      // The 2019 catechism course numbered its lessons corpus-wide; the number
      // still orders them correctly inside their CFW chapter.
      expect(parse("01-09-2019 - Aula 87 - CFW 23 - Magistrado civil").part).toBe(87);
    });

    it("reads a roman or arabic 'Parte'", () => {
      expect(parse("06-10-2024 - EBD - CFW 5 - Da providência - Parte III").part).toBe(3);
      expect(parse("15-09-2024 - EBD - CFW 7 - Do pacto - Parte II").part).toBe(2);
    });

    it("reads the parenthesised suffix", () => {
      expect(parse("08-12-2019 - O Quarto Mandamento (1)").part).toBe(1);
      expect(parse("25-09-2022 - Aplicações da Cristologia (pt.2)").part).toBe(2);
    });

    it("reads a trailing roman numeral", () => {
      expect(parse("11-06-2023 - EBD - Ensinando no caminho, ensinando na palavra II").part).toBe(
        2,
      );
    });

    it("is null when the title carries no marker", () => {
      expect(parse("28-04-2024 - Efésios 6.1-3 - Filhos").part).toBeNull();
    });

    it("does not read a chapter or verse as a part", () => {
      expect(parse("24-03-2024 - Apocalipse 2.1-7").part).toBeNull();
      expect(parse("13-10-2019 - Jonas 3").part).toBeNull();
    });
  });

  describe("display title", () => {
    it("drops the leading date", () => {
      expect(parse("17-03-2024 - Efésios 5.22-33 - O casamento diante da Cruz").displayTitle).toBe(
        "Efésios 5.22-33 - O casamento diante da Cruz",
      );
    });

    it("drops the event prefix and the date behind it", () => {
      expect(parse("IV Conferência Peregrinos_24-09-2022 - Cristologia").displayTitle).toBe(
        "Cristologia",
      );
    });

    it("tolerates the malformed dates in the corpus", () => {
      // Every one of these is a real row: stray spaces, slashes, a 5-digit year.
      expect(parse("12 -09-2021 - EBD - Presbíteros - Aula 4").displayTitle).toBe(
        "EBD - Presbíteros - Aula 4",
      );
      expect(parse("30/06/2019 - Aula 78 - CFW 28 - Batismo infantil").displayTitle).toBe(
        "Aula 78 - CFW 28 - Batismo infantil",
      );
      expect(parse('07-05-20223 - Efésios 3.7-8 - "O evangelho"').displayTitle).toBe(
        'Efésios 3.7-8 - "O evangelho"',
      );
      expect(parse("22-12-2019  - O Sexto Mandamento (1)").displayTitle).toBe(
        "O Sexto Mandamento (1)",
      );
    });

    it("leaves an undated title alone", () => {
      expect(parse("I João 3.19-24").displayTitle).toBe("I João 3.19-24");
      expect(parse("O Primeiro Mandamento").displayTitle).toBe("O Primeiro Mandamento");
    });
  });

  describe("scripture", () => {
    it("carries the reference through", () => {
      expect(parse("17-03-2024 - Efésios 5.22-33 - O casamento").scripture).toMatchObject({
        bookSlug: "efesios",
        chapterStart: 5,
      });
    });

    it("is null for a catechetical title", () => {
      expect(parse("11-10-2020 - ED - O nono mandamento").scripture).toBeNull();
    });
  });
});
