/**
 * review-log.mjs
 *
 * Shared accessor for `.quality-gate/review-log.jsonl` — the append-only log
 * of Sonnet-reviewer verdicts. Owns the path, the JSONL format, and the
 * read/append primitives used by `security-review.mjs` (writer) and the readers
 * (`pr-comment-review.mjs`, `print-quality-delta.mjs`, `show-review-log.mjs`).
 *
 * Append-only, and nothing here rewrites it. There used to be a whole-file
 * rewriter, for a post-commit hook that guessed which commit an entry belonged
 * to; it guessed wrong on 36 of 37 entries. `reviewedCommit()` records the hash
 * while it is still known, and the history it got wrong is left as written —
 * `reviewCoversCommit()` is how a reader declines to believe it.
 *
 * Schema is permissive on purpose — the log is a forensic trail, not a
 * contract. Old entries must keep parsing forever.
 */

import { execSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const repoRoot = resolve(__dirname, "..", "..");

/**
 * @typedef {Object} ReviewLogEntry
 * @property {string} ts                ISO timestamp, written when the review ran.
 * @property {string} commit            Hash of the commit the review read — see reviewedCommit().
 * @property {string} verdict           "approve" | "reject" (other values tolerated).
 * @property {string[]} [sensitiveFiles] Subset of stagedFiles flagged as sensitive paths.
 * @property {string[]} [stagedFiles]   Files staged at review time.
 * @property {string} [justification]   Reviewer's 1-3 sentence explanation.
 * @property {string[]} [concerns]      Individual concerns; empty/absent on approve.
 * @property {Object[]} [findings]      Structured issues: {severity: "blocker"|"important"|"minor", file, issue, fix}. Empty/absent on approve.
 * @property {string} [rawOutput]       Truncated raw reviewer output when verdict was unparseable.
 * @property {string} [why]             Haiku-resolved "why this PR exists" paragraph (Portuguese, 2-3 sentences) or the sentinel "Origem não clara — favor revisar manualmente". Read back by `pr-comment-review.mjs` to render the "## O que este PR resolve" lead section.
 */

export const REVIEW_LOG_PATH = join(repoRoot, ".quality-gate", "review-log.jsonl");

/** What `git push` sends for a branch deletion; there is nothing to review. */
const ZERO_SHA = "0".repeat(40);

/**
 * The commit an entry being written now is about.
 *
 * Stamped at write time, and that is the whole point. The reviewer runs from
 * `.husky/pre-push`, so the commits it reads already exist — `PUSH_LOCAL_SHA`
 * is the top of the range the hook parsed off stdin, and HEAD is the answer
 * when the script is run by hand.
 *
 * It used to write the literal `"(staged)"` and let `.husky/post-commit` fill
 * the hash in afterwards. That hook had already run by the time anyone pushed,
 * so every entry was claimed by the *next* commit somebody made: 36 of the 37
 * entries in the log name a commit that did not exist when the review ran, one
 * of them by 29 hours. The reader rendered them as approvals. Nothing can
 * reconstruct the right hash later, so it is recorded while it is still known.
 *
 * @returns {string}
 */
export function reviewedCommit() {
  const pushed = (process.env.PUSH_LOCAL_SHA ?? "").trim();
  if (pushed && pushed !== ZERO_SHA) return pushed;
  try {
    return execSync("git rev-parse HEAD", { cwd: repoRoot, encoding: "utf8" }).trim();
  } catch {
    return "(unknown)";
  }
}

/**
 * Whether an entry could actually have read the commit it names.
 *
 * A review that ran before its commit existed did not review it, whatever the
 * `commit` field says. This is the check that turns the pre-backfill entries
 * from silent green verdicts into visible warnings without rewriting the log,
 * which is forensic and stays as written.
 *
 * @param {ReviewLogEntry} entry
 * @param {string} commitIso  The commit's committer date (`git show -s --format=%cI`).
 * @returns {boolean}
 */
export function reviewCoversCommit(entry, commitIso) {
  const reviewedAt = new Date(entry?.ts ?? "").getTime();
  const committedAt = new Date(commitIso ?? "").getTime();
  if (Number.isNaN(reviewedAt) || Number.isNaN(committedAt)) return false;
  return reviewedAt >= committedAt;
}

/**
 * Load every parseable entry, in file order (oldest first).
 * Unparseable lines are skipped with a warning to stderr.
 *
 * @returns {ReviewLogEntry[]}
 */
export function loadAllEntries() {
  if (!existsSync(REVIEW_LOG_PATH)) return [];
  const raw = readFileSync(REVIEW_LOG_PATH, "utf8");
  const lines = raw.split("\n").map((l) => l.trim()).filter(Boolean);
  const entries = [];
  for (const line of lines) {
    try {
      entries.push(JSON.parse(line));
    } catch {
      console.warn(
        `[review-log] skipping unparseable line: ${line.slice(0, 120)}${line.length > 120 ? "…" : ""}`,
      );
    }
  }
  return entries;
}

/**
 * Most recent parseable entry, or null when the log is missing/empty.
 *
 * @returns {ReviewLogEntry | null}
 */
export function loadLatestEntry() {
  const entries = loadAllEntries();
  if (entries.length === 0) return null;
  return entries[entries.length - 1];
}

/**
 * Append a single entry as one JSON line. Creates the parent directory if
 * needed.
 *
 * @param {ReviewLogEntry} entry
 */
export function appendEntry(entry) {
  const dir = dirname(REVIEW_LOG_PATH);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  appendFileSync(REVIEW_LOG_PATH, `${JSON.stringify(entry)}\n`, "utf8");
}

