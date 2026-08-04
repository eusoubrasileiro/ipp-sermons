#!/usr/bin/env node
/**
 * pr-comment-review.mjs
 *
 * Wrapper around `gh pr create` for dispatched sub-agents. After ensuring a
 * PR exists for the current branch (creating one with the passthrough args
 * when missing), it reads the latest entry of .quality-gate/review-log.jsonl
 * and the deltas from .quality-gate/report.json, renders a marker-tagged
 * markdown comment, and posts (or updates) it on the PR via `gh api`. The
 * comment carries the Sonnet reviewer's APPROVE verdict + reasoning + a
 * compact quality-gate snapshot so the leader can read the verdict directly
 * on the PR page. Best-effort: any failure after the PR exists is logged as
 * a warning but never blocks the agent's flow. Refuses to run on `main`.
 */

import { execSync, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { WHY_SENTINEL } from "./lib/intent.mjs";
import { loadLatestEntry } from "./lib/review-log.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const repoRoot = resolve(__dirname, "..");

const QG_DIR = join(repoRoot, ".quality-gate");
const REPORT_PATH = join(QG_DIR, "report.json");
const COMMENT_MARKER = "<!-- ipp-sermons-reviewer-bot -->";

const RED = "\x1b[31m";
const YELLOW = "\x1b[33m";
const GREEN = "\x1b[32m";
const RESET = "\x1b[0m";

function safeExec(cmd, opts = {}) {
  try {
    return execSync(cmd, { encoding: "utf8", cwd: repoRoot, stdio: ["ignore", "pipe", "pipe"], ...opts }).trim();
  } catch {
    return "";
  }
}

function execOrThrow(cmd, opts = {}) {
  return execSync(cmd, { encoding: "utf8", cwd: repoRoot, ...opts }).trim();
}

function getCurrentBranch() {
  return safeExec("git rev-parse --abbrev-ref HEAD");
}

function getNameWithOwner() {
  const out = safeExec("gh repo view --json nameWithOwner -q .nameWithOwner");
  if (!out.includes("/")) return null;
  const [owner, repo] = out.split("/");
  return { owner, repo };
}

function getExistingPr() {
  // gh pr view exits non-zero when no PR exists for the branch.
  try {
    const out = execSync("gh pr view --json url,number", {
      encoding: "utf8",
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
    }).trim();
    if (!out) return null;
    const parsed = JSON.parse(out);
    if (parsed && typeof parsed.number === "number" && typeof parsed.url === "string") {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

function createPrWithArgs(passthroughArgs) {
  const result = spawnSync("gh", ["pr", "create", ...passthroughArgs], {
    encoding: "utf8",
    cwd: repoRoot,
    stdio: ["inherit", "pipe", "inherit"],
  });
  if (result.status !== 0) {
    process.stderr.write(`${RED}[pr-comment-review] gh pr create failed (exit ${result.status})${RESET}\n`);
    process.exit(result.status ?? 1);
  }
  // gh prints the PR URL on the last non-empty stdout line.
  const stdout = (result.stdout || "").trim();
  const lines = stdout.split("\n").map((l) => l.trim()).filter(Boolean);
  const url = lines.reverse().find((l) => l.startsWith("https://")) || stdout;
  // Echo the gh stdout so callers still see the URL printed.
  if (stdout) process.stdout.write(`${stdout}\n`);
  return url;
}

function loadReport() {
  if (!existsSync(REPORT_PATH)) return null;
  try {
    return JSON.parse(readFileSync(REPORT_PATH, "utf8"));
  } catch {
    return null;
  }
}

function formatDiff(diff) {
  if (typeof diff !== "number") {
    const n = Number(diff);
    if (Number.isFinite(n)) return formatDiff(n);
    return String(diff);
  }
  if (diff === 0) return "=";
  const rounded = Number.isInteger(diff) ? diff : Number(diff.toFixed(2));
  if (rounded === 0) return "=";
  return rounded > 0 ? `+${rounded}` : String(rounded);
}

function statusIcon(status) {
  if (status === "REGRESSION") return "✗";
  if (status === "IMPROVED") return "✓";
  return "=";
}

function trendArrow(status) {
  if (status === "REGRESSION") return " ↓";
  if (status === "IMPROVED") return " ↑";
  return "";
}

function metricCell(report, key) {
  const current = report.metrics?.[key];
  const status = report.deltas?.[key]?.status;
  return `${current ?? "—"}${trendArrow(status)}`;
}

const HEALTH_METRIC_ORDER = [
  "maxFileLinesBackend",
  "maxFileLinesFrontend",
  "filesOver500",
  "complexityViolations",
  "jscpdClones",
  "jscpdPercentage",
  "anyCasts",
  "tsSuppressors",
  "unusedExports",
  "unusedTypes",
  "unusedFiles",
  "unusedDependencies",
  "testAssertions",
];

function renderChangedTable(report) {
  const header = "| Metric | Baseline → Current | Δ | Status |\n|---|---|---|---|";
  if (!report || !report.deltas || typeof report.deltas !== "object") {
    return `${header}\n| _(no quality-gate report)_ | | | |`;
  }
  const rows = Object.entries(report.deltas)
    .filter(([, v]) => v && typeof v === "object" && v.diff !== 0)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(
      ([metric, v]) =>
        `| ${metric} | ${v.baseline} → ${v.current} | ${formatDiff(v.diff)} | ${statusIcon(v.status)} |`,
    );
  if (rows.length === 0) return `${header}\n| _(no changes)_ | | | |`;
  return `${header}\n${rows.join("\n")}`;
}

function renderCoverageTable(report) {
  const header =
    "| Layer | Stmts | Branches | Funcs | Lines |\n|---|---|---|---|---|";
  if (!report || !report.metrics) return `${header}\n| _(no report)_ | | | | |`;
  const backend = `| Backend  | ${metricCell(report, "coverageBackendStatements")} | ${metricCell(report, "coverageBackendBranches")} | ${metricCell(report, "coverageBackendFunctions")} | ${metricCell(report, "coverageBackendLines")} |`;
  const frontend = `| Frontend | ${metricCell(report, "coverageFrontendStatements")} | ${metricCell(report, "coverageFrontendBranches")} | ${metricCell(report, "coverageFrontendFunctions")} | ${metricCell(report, "coverageFrontendLines")} |`;
  return `${header}\n${backend}\n${frontend}`;
}

function renderHealthTable(report) {
  const header = "| Metric | Baseline | Current | Δ |\n|---|---|---|---|";
  if (!report || !report.metrics) return `${header}\n| _(no report)_ | | | |`;
  const rows = HEALTH_METRIC_ORDER.map((key) => {
    const current = report.metrics[key];
    const delta = report.deltas?.[key];
    const baseline = delta?.baseline ?? report.baseline?.[key] ?? "—";
    let diffDisplay = "—";
    if (delta && typeof delta.diff !== "undefined") {
      diffDisplay = formatDiff(delta.diff);
    } else if (baseline !== "—" && current !== undefined) {
      const baseNum = Number(baseline);
      const curNum = Number(current);
      if (Number.isFinite(baseNum) && Number.isFinite(curNum)) {
        diffDisplay = formatDiff(curNum - baseNum);
      }
    }
    const arrow = trendArrow(delta?.status);
    return `| ${key} | ${baseline} | ${current ?? "—"} | ${diffDisplay}${arrow} |`;
  });
  return `${header}\n${rows.join("\n")}`;
}

function renderConcernsBlock(entry) {
  const concerns = entry && Array.isArray(entry.concerns) ? entry.concerns : [];
  if (concerns.length === 0) return "";
  const lines = concerns.map((c) => `> - ${c}`).join("\n");
  return `> **⚠ Concerns**\n${lines}\n\n`;
}

function renderSensitiveBlock(entry) {
  const files = entry && Array.isArray(entry.sensitiveFiles) ? entry.sensitiveFiles : [];
  if (files.length === 0) return "";
  return `> **🔒 Sensitive files touched:** ${files.join(", ")}\n\n`;
}

function renderFindingsBlock(entry) {
  const findings = entry && Array.isArray(entry.findings) ? entry.findings : [];
  if (findings.length === 0) return "";
  const header = "| Severity | File | Issue | Fix |\n|---|---|---|---|";
  const rows = findings
    .map((f) => `| ${f.severity} | \`${f.file}\` | ${f.issue} | ${f.fix} |`)
    .join("\n");
  return `**Findings**\n\n${header}\n${rows}\n\n`;
}

function renderTddBlock() {
  try {
    const out = safeExec("git diff --name-status main...HEAD");
    if (!out) return "✅ No new source files in this PR.";
    const added = out
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .filter((line) => line.startsWith("A\t"))
      .map((line) => line.slice(2));

    const sourceCandidates = added.filter((path) => {
      if (!/^(backend|frontend)\/src\//.test(path)) return false;
      if (path.endsWith(".test.ts") || path.endsWith(".test.tsx")) return false;
      if (path.endsWith(".spec.ts") || path.endsWith(".spec.tsx")) return false;
      if (path.endsWith(".d.ts")) return false;
      if (/\/__tests__\//.test(path)) return false;
      if (/\/mocks\//.test(path)) return false;
      if (!/\.(ts|tsx)$/.test(path)) return false;
      return true;
    });

    if (sourceCandidates.length === 0) {
      return "✅ No new source files in this PR.";
    }

    const missing = sourceCandidates.filter((path) => {
      const dot = path.lastIndexOf(".");
      const stem = path.slice(0, dot);
      const ext = path.slice(dot);
      const sibling = `${stem}.test${ext}`;
      if (added.includes(sibling)) return false;
      // Sibling might pre-exist on disk (rare but valid).
      if (existsSync(join(repoRoot, sibling))) return false;
      return true;
    });

    if (missing.length === 0) {
      return "✅ All new source files under `src/**` have a sibling `*.test.ts`.";
    }
    const list = missing.map((p) => `- ${p}`).join("\n");
    return `⚠ New source files missing tests:\n${list}`;
  } catch {
    return "_(TDD check skipped)_";
  }
}

function renderQualityGateHeader(report) {
  if (!report) return "### Quality Gate — _(report not found — run `pnpm quality-gate`)_";
  const overall = report.overall ?? "?";
  const regressions = report.regressions ?? 0;
  const improvements = report.improvements ?? 0;
  const regWord = regressions === 1 ? "regression" : "regressions";
  const impWord = improvements === 1 ? "improvement" : "improvements";
  return `### Quality Gate — ${overall} · ${regressions} ${regWord} · ${improvements} ${impWord}`;
}

function getFilesCount(entry) {
  if (entry && Array.isArray(entry.stagedFiles) && entry.stagedFiles.length > 0) {
    return entry.stagedFiles.length;
  }
  const out = safeExec("git diff --name-only main...HEAD");
  if (!out) return 0;
  return out.split("\n").filter(Boolean).length;
}

function buildBody(entry, report, branch) {
  const justification = (entry && entry.justification) || "(no justification recorded)";
  const why = (entry && entry.why) || WHY_SENTINEL;
  const concernsBlock = renderConcernsBlock(entry);
  const findingsBlock = renderFindingsBlock(entry);
  const sensitiveBlock = renderSensitiveBlock(entry);
  const qgHeader = renderQualityGateHeader(report);
  const reportMissing = !report;
  const filesCount = getFilesCount(entry);
  const ts = (entry && entry.ts) || new Date().toISOString();

  const tablesSection = reportMissing
    ? "_(quality-gate report not found — run `pnpm quality-gate`)_"
    : `**Changed metrics**

${renderChangedTable(report)}

**Coverage**

${renderCoverageTable(report)}

**Code Health** (lower = better, except testAssertions)

${renderHealthTable(report)}`;

  return `${COMMENT_MARKER}
## O que este PR resolve

${why}

## Sonnet Reviewer — APPROVE

**Justification:** ${justification}

${concernsBlock}${findingsBlock}${sensitiveBlock}${qgHeader}

${tablesSection}

### TDD Discipline

${renderTddBlock()}

<sub>Reviewed at ${ts} · branch \`${branch}\` · ${filesCount} files staged</sub>
`;
}

function findExistingComment(owner, repo, prNumber) {
  const result = spawnSync(
    "gh",
    [
      "api",
      `repos/${owner}/${repo}/issues/${prNumber}/comments`,
      "--paginate",
      "--jq",
      `.[] | select(.body | startswith("${COMMENT_MARKER}")) | .id`,
    ],
    { encoding: "utf8", cwd: repoRoot, stdio: ["ignore", "pipe", "pipe"] },
  );
  if (result.status !== 0 || !result.stdout) return null;
  const ids = result.stdout.trim().split("\n").map((l) => l.trim()).filter(Boolean);
  if (ids.length === 0) return null;
  return ids[ids.length - 1]; // newest/last match
}

function writeBodyToTempFile(body) {
  const dir = mkdtempSync(join(tmpdir(), "pr-comment-review-"));
  const path = join(dir, "body.md");
  writeFileSync(path, body, "utf8");
  return { path, dir };
}

function postOrUpdateComment(owner, repo, prNumber, body) {
  const existingId = findExistingComment(owner, repo, prNumber);
  const { path: bodyPath, dir: tmpDir } = writeBodyToTempFile(body);
  try {
    if (existingId) {
      const result = spawnSync(
        "gh",
        [
          "api",
          "--method",
          "PATCH",
          `repos/${owner}/${repo}/issues/comments/${existingId}`,
          "-F",
          `body=@${bodyPath}`,
        ],
        { encoding: "utf8", cwd: repoRoot, stdio: ["ignore", "pipe", "pipe"] },
      );
      if (result.status !== 0) {
        process.stderr.write(
          `${YELLOW}[pr-comment-review] failed to update existing comment ${existingId}: ${result.stderr || result.stdout}${RESET}\n`,
        );
        return false;
      }
      process.stdout.write(`${GREEN}[pr-comment-review] updated reviewer comment on PR #${prNumber}${RESET}\n`);
      return true;
    }
    const result = spawnSync(
      "gh",
      [
        "api",
        "--method",
        "POST",
        `repos/${owner}/${repo}/issues/${prNumber}/comments`,
        "-F",
        `body=@${bodyPath}`,
      ],
      { encoding: "utf8", cwd: repoRoot, stdio: ["ignore", "pipe", "pipe"] },
    );
    if (result.status !== 0) {
      process.stderr.write(
        `${YELLOW}[pr-comment-review] failed to post comment: ${result.stderr || result.stdout}${RESET}\n`,
      );
      return false;
    }
    process.stdout.write(`${GREEN}[pr-comment-review] posted reviewer comment on PR #${prNumber}${RESET}\n`);
    return true;
  } finally {
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {
      // ignore
    }
  }
}

function main() {
  const rawArgs = process.argv.slice(2);
  const dryRun = rawArgs.includes("--dry");
  const passthroughArgs = rawArgs.filter((a) => a !== "--dry");

  const branch = getCurrentBranch();
  if (!branch) {
    process.stderr.write(`${RED}[pr-comment-review] could not determine current branch${RESET}\n`);
    process.exit(1);
  }
  if (branch === "main" && !dryRun) {
    process.stderr.write(
      `${RED}[pr-comment-review] refusing to run on main — checkout a feature branch first${RESET}\n`,
    );
    process.exit(1);
  }

  if (dryRun) {
    const entry = loadLatestEntry();
    const report = loadReport();
    process.stdout.write(buildBody(entry, report, branch));
    process.exit(0);
  }

  let prUrl;
  let prNumber;
  const existing = getExistingPr();
  if (existing) {
    prUrl = existing.url;
    prNumber = existing.number;
    process.stdout.write(`[pr-comment-review] PR already exists: ${prUrl}\n`);
  } else {
    prUrl = createPrWithArgs(passthroughArgs);
    const after = getExistingPr();
    if (!after) {
      process.stderr.write(
        `${YELLOW}[pr-comment-review] PR created but could not resolve number; skipping comment${RESET}\n`,
      );
      process.stdout.write(`${prUrl}\n`);
      process.exit(0);
    }
    prNumber = after.number;
    if (!prUrl) prUrl = after.url;
  }

  // Best-effort comment posting from here on. Never block.
  try {
    const entry = loadLatestEntry();
    if (!entry) {
      process.stdout.write(
        `[pr-comment-review] no review-log entry found; skipping comment.\n`,
      );
      process.stdout.write(`${prUrl}\n`);
      process.exit(0);
    }
    if (entry.verdict !== "approve") {
      process.stdout.write(
        `[pr-comment-review] latest review verdict is "${entry.verdict}"; skipping comment.\n`,
      );
      process.stdout.write(`${prUrl}\n`);
      process.exit(0);
    }

    const repoSlug = getNameWithOwner();
    if (!repoSlug) {
      process.stderr.write(
        `${YELLOW}[pr-comment-review] could not resolve repo owner/name; skipping comment${RESET}\n`,
      );
      process.stdout.write(`${prUrl}\n`);
      process.exit(0);
    }

    const report = loadReport();
    const body = buildBody(entry, report, branch);
    postOrUpdateComment(repoSlug.owner, repoSlug.repo, prNumber, body);
  } catch (err) {
    process.stderr.write(
      `${YELLOW}[pr-comment-review] unexpected error while posting comment (PR is fine): ${err && err.message ? err.message : err}${RESET}\n`,
    );
  }

  // Final stdout line is the PR URL so the agent's "report back" still works.
  process.stdout.write(`${prUrl}\n`);
  process.exit(0);
}

main();
