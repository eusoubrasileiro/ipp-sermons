import { describe, expect, it } from "vitest";
import { verdict } from "../../src/lib/facets/verdict.ts";

/**
 * `verify:facets` has three outcomes, not two.
 *
 * A new series name that canonicalisation has never seen is the one thing a
 * human is meant to look at after adding sermons — and it is not an error: the
 * sermons load fine, they just browse without a series. Collapsing it into
 * exit 0 left it detectable only by grepping a Portuguese sentence out of the
 * stage's own stdout, which is not something an orchestrator can be built on.
 */
const erro = { severity: "erro" as const, message: "livro desconhecido" };
const aviso = { severity: "aviso" as const, message: "série nova" };

describe("verdict", () => {
  it("passes clean", () => {
    expect(verdict([])).toMatchObject({ exitCode: 0, errors: [], warnings: [] });
  });

  it("exits 2 when there are only warnings", () => {
    expect(verdict([aviso])).toMatchObject({ exitCode: 2, warnings: [aviso] });
  });

  it("exits 1 when there is any error", () => {
    expect(verdict([erro])).toMatchObject({ exitCode: 1, errors: [erro] });
  });

  it("lets an error outrank a warning", () => {
    // Otherwise a run with both would report the softer outcome, and a caller
    // that only stops on 1 would sail past a broken corpus.
    expect(verdict([aviso, erro]).exitCode).toBe(1);
  });

  it("keeps the two severities apart", () => {
    expect(verdict([aviso, erro, aviso])).toMatchObject({
      errors: [erro],
      warnings: [aviso, aviso],
    });
  });
});
