#!/usr/bin/env node
/**
 * show-review-log.mjs
 *
 * For each commit in `git log origin/main..HEAD`, prints:
 *   - hash + subject
 *   - modified files grouped by category
 *   - reviewer verdict + justification (from .quality-gate/review-log.jsonl)
 *   - WARNING if no review record found, or if the one found predates the commit
 *
 * Always exits 0 (informational). Human aborts with Ctrl+C if anything looks wrong.
 */

import { execSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { loadAllEntries, reviewCoversCommit } from "./lib/review-log.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const repoRoot = resolve(__dirname, "..");

const RED = "\x1b[31m";
const YELLOW = "\x1b[33m";
const GREEN = "\x1b[32m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

function safeExec(cmd) {
  try {
    return execSync(cmd, { encoding: "utf8", cwd: repoRoot });
  } catch (err) {
    return err.stdout?.toString() ?? "";
  }
}

function categorize(file) {
  if (file.startsWith("backend/test/e2e/real/") || file.startsWith("frontend/tests/e2e-real/"))
    return "E2E-REAL";
  if (file.startsWith("backend/test/") || file.startsWith("frontend/tests/")) return "TEST";
  if (file.startsWith(".husky/")) return "HOOK";
  if (
    file === "playwright.config.ts" ||
    file === "vitest.e2e.config.ts" ||
    file === ".claude/settings.json" ||
    file.endsWith("/playwright.config.ts") ||
    file.endsWith("/vitest.e2e.config.ts")
  )
    return "CONFIG";
  if (file === "quality-baseline.json") return "BASELINE";
  return "SOURCE";
}

function getCommits() {
  // origin/main may not exist (e.g. no remote, fresh worktree). Fall back gracefully.
  let raw = safeExec("git log --oneline origin/main..HEAD 2>/dev/null");
  if (!raw.trim()) {
    raw = safeExec("git log --oneline -5");
  }
  return raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((line) => {
      const idx = line.indexOf(" ");
      return idx === -1
        ? { hash: line, subject: "" }
        : { hash: line.slice(0, idx), subject: line.slice(idx + 1) };
    });
}

function getCommitFiles(hash) {
  const out = safeExec(`git show --name-only --pretty=format: ${hash}`);
  return out
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

function getCommitTimestamp(hash) {
  return safeExec(`git show -s --format=%cI ${hash}`).trim();
}

/**
 * The review of `hash`, if one can honestly be said to exist.
 *
 * Returns `{ entry }` when the log holds a review that named this commit and
 * ran after it, `{ stale }` when it named it but could not have read it, and
 * `{}` when nothing named it at all. The caller renders the three differently,
 * because they are three different things to be told at a push.
 *
 * Hash only. This used to fall back to matching an entry by file set and
 * nearest timestamp, which existed because entries carried `"(staged)"` rather
 * than a hash and had to be correlated by coincidence. `reviewedCommit()`
 * removed the reason, and matching on a coincidence of file names is one of the
 * ways a review of one push ends up displayed against a different commit.
 */
function findReviewEntryForCommit(reviewLog, hash) {
  const committedAt = getCommitTimestamp(hash);
  const named = reviewLog.filter((e) => e.commit === hash);
  const covering = named.filter((e) => reviewCoversCommit(e, committedAt));
  if (covering.length > 0) return { entry: covering[covering.length - 1] };
  if (named.length > 0) return { stale: named[named.length - 1] };
  return {};
}

function main() {
  const commits = getCommits();
  if (commits.length === 0) {
    process.stdout.write(`${DIM}(no commits to display)${RESET}\n`);
    process.exit(0);
  }

  const reviewLog = loadAllEntries();

  for (const { hash, subject } of commits) {
    process.stdout.write(`\n${BOLD}${hash}${RESET} ${subject}\n`);

    const files = getCommitFiles(hash);
    if (files.length === 0) {
      process.stdout.write(`  ${DIM}(no files)${RESET}\n`);
      continue;
    }

    const grouped = new Map();
    for (const f of files) {
      const cat = categorize(f);
      if (!grouped.has(cat)) grouped.set(cat, []);
      grouped.get(cat).push(f);
    }

    const order = ["E2E-REAL", "HOOK", "CONFIG", "BASELINE", "TEST", "SOURCE"];
    for (const cat of order) {
      if (!grouped.has(cat)) continue;
      const tagColor =
        cat === "E2E-REAL" || cat === "HOOK" || cat === "CONFIG"
          ? RED
          : cat === "BASELINE" || cat === "TEST"
            ? YELLOW
            : DIM;
      for (const f of grouped.get(cat)) {
        process.stdout.write(`  ${tagColor}[${cat}]${RESET} ${f}\n`);
      }
    }

    const { entry, stale } = findReviewEntryForCommit(reviewLog, hash);
    if (stale) {
      // Every entry written before reviewedCommit() landed: the reviewer ran at
      // pre-push and stamped "(staged)", and `.husky/post-commit` handed that
      // entry to whichever commit came next. The verdict below it is real, but
      // it is a verdict on some other push, so it is not shown as one.
      process.stdout.write(
        `  ${RED}${BOLD}⚠ REVIEW RECORD PREDATES COMMIT${RESET} ` +
          `${DIM}(reviewed ${stale.ts}, committed ${getCommitTimestamp(hash)})${RESET}\n`,
      );
      continue;
    }
    if (!entry) {
      process.stdout.write(`  ${RED}${BOLD}⚠ NO REVIEW RECORD${RESET}\n`);
      continue;
    }

    if (entry.verdict === "reject" || entry.verdict === "escalate") {
      const label = entry.verdict === "reject" ? "REJECT" : "ESCALATE";
      process.stdout.write(`  ${RED}${BOLD}Verdict: ${label}${RESET}\n`);
      process.stdout.write(`  ${RED}Justification:${RESET} ${entry.justification}\n`);
      if (Array.isArray(entry.concerns) && entry.concerns.length > 0) {
        process.stdout.write(`  ${RED}Concerns:${RESET}\n`);
        for (const c of entry.concerns) {
          process.stdout.write(`    ${RED}- ${c}${RESET}\n`);
        }
      }
      if (Array.isArray(entry.findings) && entry.findings.length > 0) {
        process.stdout.write(`  ${RED}Findings:${RESET}\n`);
        for (const f of entry.findings) {
          const idTag = f.id ? `[${f.id}]` : "";
          process.stdout.write(
            `    ${RED}${idTag}[${f.severity}] ${f.file} — ${f.issue} → ${f.fix}${RESET}\n`,
          );
        }
      }
    } else {
      process.stdout.write(`  ${GREEN}Verdict: approve${RESET} — ${entry.justification}\n`);
    }
  }

  process.exit(0);
}

main();
