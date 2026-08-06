/**
 * Deterministic critical-path ratification (standards.md §6, "Ratified-by trailer").
 *
 * The critical paths in this repo — the migration SQL, the deploy topology,
 * the corpus, the golden query contract — are things an agent must not change
 * on its own judgement. But they are perfectly legitimate agent work *after*
 * the owner ratifies it, and a reviewer running in a pre-push hook cannot see
 * an in-session ratification.
 *
 * The machine-visible marker is a `Ratified-by:` commit trailer. Presence is
 * computed here from git and handed to the model as a fact — never inferred by
 * the model from commit prose, which would make the marker forgeable by
 * writing the right sentence in a commit body.
 *
 * It is an audit marker of an explicit ratification event, not proof of one —
 * the same trust model as the rest of the harness. What it buys is that
 * ratified work stops being a reason to reach for `--no-verify`, which is the
 * only outcome worse than an over-eager reject.
 */

/**
 * Mirrors the critical-paths list in the reviewer prompt, the `ask` tier in
 * `.claude/settings.json`, and the Critical files table in CLAUDE.md. All four
 * have to say the same thing — see CLAUDE.md, "Critical files".
 */
export const CRITICAL_PATHS = [
  /^backend\/prisma\//,
  /^backend\/scripts\/db-push\.sh$/,
  /^deploy\//,
  /^Dockerfile$/,
  /^docker-compose\.yml$/,
  /^data\//,
  /^backend\/test\/golden\/queries\.json$/,
  /^\.gitignore$/,
  /\.md$/,
];

export function isCriticalPath(file) {
  return CRITICAL_PATHS.some((re) => re.test(file));
}

/**
 * Splits the push range into the commits that need a trailer and the ones that
 * carry it.
 *
 * @param {{sha: string, subject: string, files: string[], ratifiedBy: string}[]} commits
 * @returns {{needsRatification: object[], missing: object[], allRatified: boolean}}
 */
export function ratificationFacts(commits) {
  const needsRatification = commits
    .map((c) => ({ ...c, criticalFiles: c.files.filter(isCriticalPath) }))
    .filter((c) => c.criticalFiles.length > 0);

  const missing = needsRatification.filter((c) => !c.ratifiedBy?.trim());

  return { needsRatification, missing, allRatified: missing.length === 0 };
}

/**
 * Renders the facts as the prompt section the model is told to trust over its
 * own reading of the diff.
 */
export function ratificationSection(facts) {
  if (facts.needsRatification.length === 0) {
    return "No commit in this push range touches a critical path.";
  }

  const lines = facts.needsRatification.map((c) => {
    const mark = c.ratifiedBy?.trim() ? `RATIFIED by ${c.ratifiedBy.trim()}` : "NOT RATIFIED";
    const files = c.criticalFiles.slice(0, 6).join(", ");
    const more = c.criticalFiles.length > 6 ? ` (+${c.criticalFiles.length - 6} more)` : "";
    return `- ${c.sha.slice(0, 7)} ${c.subject} — ${mark}\n    ${files}${more}`;
  });

  const verdict = facts.allRatified
    ? "Every commit touching a critical path carries a `Ratified-by` trailer."
    : `${facts.missing.length} commit(s) touch a critical path WITHOUT a \`Ratified-by\` trailer.`;

  return `${verdict}\n\n${lines.join("\n")}`;
}
