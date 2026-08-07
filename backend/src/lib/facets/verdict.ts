/**
 * The three outcomes of a facet verification.
 *
 * Errors block: an unknown book slug or a chapter past the end of its book is a
 * corpus that must not be loaded. Warnings do not: a series name that
 * canonicalisation has never seen still loads, the sermons just browse without
 * a series until someone runs `canonicalize:series`.
 *
 * That middle case is the whole reason this is a separate exit code. It is the
 * one thing a human is meant to look at after adding sermons, and an
 * orchestrator can only branch on it if it is a number rather than a Portuguese
 * sentence somewhere in stdout.
 */
export type Problem = { severity: "erro" | "aviso"; message: string };

export function verdict(problems: Problem[]): {
  errors: Problem[];
  warnings: Problem[];
  exitCode: 0 | 1 | 2;
} {
  const errors = problems.filter((p) => p.severity === "erro");
  const warnings = problems.filter((p) => p.severity === "aviso");
  // An error outranks a warning: a caller that only stops on 1 must not sail
  // past a broken corpus because a new series happened to show up too.
  const exitCode = errors.length > 0 ? 1 : warnings.length > 0 ? 2 : 0;

  return { errors, warnings, exitCode };
}
