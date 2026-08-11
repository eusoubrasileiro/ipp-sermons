import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

/**
 * Who reviewed what — the only record that a push was ever looked at.
 *
 * `.quality-gate/review-log.jsonl` is the forensic trail behind the pre-push
 * reviewer, and for most of its life it named the wrong commit. The reviewer
 * runs at pre-push and wrote `commit: "(staged)"`; `.husky/post-commit` filled
 * that in with HEAD — but post-commit has already run by the time anyone
 * pushes, so each entry was claimed by the NEXT commit somebody made. 36 of 37
 * entries named a commit that did not exist when the review ran, and the one
 * approving 9863596 shares zero of its nine files.
 *
 * A false record is worse than a missing one: the display renders it green.
 * These tests hold both ends — the writer names the commit it actually read,
 * and the reader refuses an entry that cannot have seen the commit it claims.
 *
 * The lib is loaded through a computed URL because it is `.mjs` outside this
 * package's rootDir; a literal specifier would not typecheck.
 */
const ROOT = join(import.meta.dirname, "../..");
const LIB = new URL("../../scripts/lib/review-log.mjs", import.meta.url).href;

const lib = await import(LIB);
const { reviewedCommit, reviewCoversCommit } = lib;

const HEAD = execSync("git rev-parse HEAD", { cwd: ROOT, encoding: "utf8" }).trim();

afterEach(() => {
  delete process.env.PUSH_LOCAL_SHA;
});

describe("reviewedCommit", () => {
  it("names the commit at the top of the range being pushed", () => {
    // The pre-push hook parses this off stdin and exports it. It is the only
    // moment anything knows which commits are under review, which is why the
    // entry has to be stamped here rather than guessed at afterwards.
    process.env.PUSH_LOCAL_SHA = "a".repeat(40);

    expect(reviewedCommit()).toBe("a".repeat(40));
  });

  it("falls back to HEAD when run outside the hook", () => {
    expect(reviewedCommit()).toBe(HEAD);
  });

  it("ignores the zero SHA a branch deletion carries", () => {
    process.env.PUSH_LOCAL_SHA = "0".repeat(40);

    expect(reviewedCommit()).toBe(HEAD);
  });

  it("never yields the placeholder that caused the misattribution", () => {
    process.env.PUSH_LOCAL_SHA = "";

    expect(reviewedCommit()).not.toBe("(staged)");
  });
});

describe("reviewCoversCommit", () => {
  const COMMIT = "2026-08-10T21:12:34-03:00";

  it("accepts a review that ran after the commit it names", () => {
    // The normal pre-push case: commit, then push a minute or an hour later.
    expect(reviewCoversCommit({ ts: "2026-08-11T00:14:06.000Z" }, COMMIT)).toBe(true);
  });

  it("accepts a review that ran at the same instant", () => {
    expect(reviewCoversCommit({ ts: "2026-08-11T00:12:34.000Z" }, COMMIT)).toBe(true);
  });

  it("refuses a review that ran before the commit existed", () => {
    // The 36 backfilled entries. This one is dated 29 hours before the commit
    // it claims to have approved, and it rendered as a green verdict.
    expect(reviewCoversCommit({ ts: "2026-08-09T18:44:33.000Z" }, COMMIT)).toBe(false);
  });

  it("refuses an entry with no usable timestamp rather than assuming", () => {
    expect(reviewCoversCommit({ ts: "not a date" }, COMMIT)).toBe(false);
    expect(reviewCoversCommit({}, COMMIT)).toBe(false);
  });

  it("refuses when the commit's own date is unreadable", () => {
    expect(reviewCoversCommit({ ts: "2026-08-11T00:14:06.000Z" }, "")).toBe(false);
  });
});

describe("the log's writers and readers", () => {
  it("leaves no writer still stamping the placeholder", () => {
    // The four call sites in main() — reviewer unavailable, unparseable
    // payload, the verdict itself, and the empty-range short circuit. Missing
    // one puts an unattributable entry back in the log on exactly the paths
    // that block a push, which are the ones an operator most needs to read.
    const src = readFileSync(join(ROOT, "scripts/security-review.mjs"), "utf8");

    expect(src).not.toMatch(/commit:\s*"\(staged\)"/);
    expect(src.match(/commit:\s*reviewedCommit\(\)/g) ?? []).toHaveLength(4);
  });

  it("no longer guesses a commit for an entry after the fact", () => {
    // `.husky/post-commit` used to rewrite the newest "(staged)" entry to HEAD.
    // With the writer stamping the real SHA there is nothing left to backfill,
    // and the guess is what produced every wrong attribution in the log.
    const hook = readFileSync(join(ROOT, ".husky/post-commit"), "utf8");

    expect(hook).not.toMatch(/backfill-review-log/);
  });

  it("keeps the reader from correlating an entry by anything but its hash", () => {
    // It used to fall back to matching an entry by file set and timestamp when
    // no hash matched. That fallback existed only because entries had no usable
    // hash, and matching on a coincidence of file names is how a review of one
    // push gets shown against a different commit.
    const src = readFileSync(join(ROOT, "scripts/show-review-log.mjs"), "utf8");

    expect(src).toMatch(/reviewCoversCommit/);
    expect(src).not.toMatch(/bestTsDiff/);
  });
});

describe("findReviewEntryForCommit", () => {
  const SHORT = HEAD.slice(0, 7);
  const COMMITTED_AT = execSync(`git show -s --format=%cI ${HEAD}`, {
    cwd: ROOT,
    encoding: "utf8",
  }).trim();
  const after = new Date(new Date(COMMITTED_AT).getTime() + 60_000).toISOString();
  const before = new Date(new Date(COMMITTED_AT).getTime() - 60_000).toISOString();

  it("matches an abbreviated hash against the full one the log records", async () => {
    // The defect this exists for: the writer records `git rev-parse` output and
    // the display listed commits with `--oneline`, so `entry.commit === hash`
    // compared 40 characters against 7 and never matched -- for the whole life
    // of the file. Nobody saw it because the file-set heuristic answered
    // instead, badly. Deleting the heuristic is what made it visible.
    const { findReviewEntryForCommit } = await import(
      new URL("../../scripts/show-review-log.mjs", import.meta.url).href
    );

    const found = findReviewEntryForCommit([{ commit: HEAD, ts: after }], SHORT);

    expect(found.entry?.ts).toBe(after);
    expect(found.stale).toBeUndefined();
  });

  it("reports a record that predates its commit as stale, not as a verdict", async () => {
    const { findReviewEntryForCommit } = await import(
      new URL("../../scripts/show-review-log.mjs", import.meta.url).href
    );

    const found = findReviewEntryForCommit(
      [{ commit: HEAD, ts: before, verdict: "approve" }],
      SHORT,
    );

    expect(found.entry).toBeUndefined();
    expect(found.stale?.ts).toBe(before);
  });

  it("reports nothing at all when no entry names the commit", async () => {
    const { findReviewEntryForCommit } = await import(
      new URL("../../scripts/show-review-log.mjs", import.meta.url).href
    );

    const found = findReviewEntryForCommit([{ commit: "b".repeat(40), ts: after }], SHORT);

    expect(found.entry).toBeUndefined();
    expect(found.stale).toBeUndefined();
  });
});
