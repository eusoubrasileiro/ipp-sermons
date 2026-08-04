#!/usr/bin/env node
/**
 * Quality Gate Script (see standards/standards.md §4)
 *
 * Measures code-quality metrics and compares them against the baseline
 * stored in quality-baseline.json. Exits 0 on pass, 1 on any regression.
 *
 * Usage:
 *   node scripts/quality-gate.mjs              # compare against baseline
 *   node scripts/quality-gate.mjs --update-baseline  # snapshot current state
 *
 * Adapted from the template for this repo's layout: three source roots
 * (`backend/src`, `frontend/src`, `shared/src`) and co-located frontend/shared
 * tests rather than the template's `frontend/tests` directory. The baseline
 * schema only has two size buckets, so `shared/` — server-authored domain
 * schemas — is counted on the backend side rather than going unmeasured.
 */

import { execSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const BASELINE_FILE = path.join(ROOT, "quality-baseline.json");
const REPORT_DIR = path.join(ROOT, ".quality-gate");
const REPORT_FILE = path.join(REPORT_DIR, "report.json");

const UPDATE_BASELINE = process.argv.includes("--update-baseline");
const NO_ATTRIBUTION = process.argv.includes("--no-attribution");
// Test/escape hatch: compare against an alternate baseline file instead of the
// committed one (never writes to it). Used by the attribution acceptance test.
const BASELINE_OVERRIDE = (() => {
  const i = process.argv.indexOf("--baseline");
  return i >= 0 ? process.argv[i + 1] : null;
})();

// ─── Helpers ─────────────────────────────────────────────────────────────────

function run(cmd, { cwd = ROOT, silent = true } = {}) {
  try {
    return execSync(cmd, {
      cwd,
      stdio: silent ? ["pipe", "pipe", "pipe"] : "inherit",
      encoding: "utf8",
    });
  } catch (err) {
    return err.stdout || "";
  }
}

function ensureDir(dir) {
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

// ─── Metric collectors ────────────────────────────────────────────────────────

function collectFileSizes(root = ROOT) {
  // Use a single find with proper grouping to avoid -o precedence issues
  const allFilesOut = run(
    `find backend/src frontend/src shared/src \\( -name '*.ts' -o -name '*.tsx' \\) -not -path '*/node_modules/*' -not -path '*/dist/*' | xargs wc -l 2>/dev/null`,
    { cwd: root },
  );

  const allLines = allFilesOut.trim().split("\n");
  let filesOver500 = 0;
  let maxBackend = 0;
  let maxFrontend = 0;

  for (const line of allLines) {
    const parts = line.trim().split(/\s+/);
    if (parts.length < 2) continue;
    const count = parseInt(parts[0]);
    const file = parts[1];
    if (!file || file === "total" || Number.isNaN(count)) continue;
    if (count > 500) filesOver500++;
    // shared/ folds into the backend bucket: it is server-authored domain code
    // and the baseline schema has no third bucket to put it in.
    if ((file.startsWith("backend/") || file.startsWith("shared/")) && count > maxBackend) {
      maxBackend = count;
    }
    if (file.startsWith("frontend/") && count > maxFrontend) maxFrontend = count;
  }

  return {
    maxFileLinesBackend: maxBackend,
    maxFileLinesFrontend: maxFrontend,
    filesOver500,
  };
}

function collectComplexity(root = ROOT) {
  const output = run(`pnpm exec biome check --reporter=json 2>/dev/null`, { cwd: root });
  try {
    const d = JSON.parse(output);
    const diags = d.diagnostics || [];
    const cc = diags.filter(
      (d) => d.category === "lint/complexity/noExcessiveCognitiveComplexity",
    );
    return { complexityViolations: cc.length };
  } catch {
    return { complexityViolations: 0 };
  }
}

// knip 5.x emits `{ files: string[], issues: PerFile[] }`: `files` is the list
// of wholly-unused files, and each `issues` entry carries per-file arrays.
// The template parsed the inverse shape (`issues.files`, `files[].exports`),
// which silently returns four zeros against a current knip — a gate reporting
// success without having run, which is the worst failure mode there is
// (standards.md §8.2). Parse defensively so a future shape change fails loudly
// via the baseline rather than quietly reading as clean.
function parseKnip(output) {
  const jsonStart = output.indexOf("{");
  if (jsonStart < 0) return null;
  let d;
  try {
    d = JSON.parse(output.slice(jsonStart));
  } catch {
    return null;
  }
  const perFile = Array.isArray(d.issues) ? d.issues : [];
  const sum = (key) => perFile.reduce((acc, f) => acc + (f[key]?.length ?? 0), 0);
  return {
    unusedExports: sum("exports"),
    unusedTypes: sum("types"),
    unusedFiles: Array.isArray(d.files) ? d.files.length : 0,
    unusedDependencies: sum("dependencies") + sum("devDependencies") + sum("unlisted"),
  };
}

function collectKnip(root = ROOT) {
  // knip exits non-zero whenever it finds anything, so `run` swallowing the
  // failure and returning stdout is the intended path, not an error case.
  const output = run(`pnpm exec knip --reporter json 2>/dev/null`, { cwd: root });
  const parsed = parseKnip(output);
  if (parsed) return parsed;
  // Unparseable means knip did not run — a bad knip.json rejects the whole
  // config and prints nothing to stdout. Returning zeros here would report four
  // clean metrics from a collector that never executed, so fail loudly instead.
  // Attribution runs collectors in a throwaway worktree and catches this,
  // marking the metric "unknown" (which still blocks).
  throw new Error(
    "knip produced no parseable JSON — the run failed. Check `pnpm exec knip --reporter json` and knip.json.",
  );
}

function collectJscpd(root = ROOT) {
  const reportPath = path.join(root, ".quality-gate", "jscpd", "jscpd-report.json");
  ensureDir(path.join(root, ".quality-gate", "jscpd"));
  run(
    `pnpm exec jscpd backend/src frontend/src shared/src --reporters json --output .quality-gate/jscpd 2>/dev/null`,
    { cwd: root },
  );
  try {
    const report = JSON.parse(readFileSync(reportPath, "utf8"));
    const clones = report.duplicates?.length ?? 0;
    const percentage = report.statistics?.total?.percentage ?? 0;
    return {
      jscpdClones: clones,
      jscpdPercentage: String(percentage.toFixed(2)),
    };
  } catch {
    return { jscpdClones: 0, jscpdPercentage: "0.00" };
  }
}

function collectAnyCasts(root = ROOT) {
  const output = run(
    `grep -rE "(: any\\b| as any\\b|<any>)" --include='*.ts' --include='*.tsx' backend/src frontend/src shared/src 2>/dev/null || true`,
    { cwd: root },
  );
  const lines = output.trim().split("\n").filter(Boolean);
  return { anyCasts: lines.length };
}

function collectTsSuppressors(root = ROOT) {
  const output = run(
    `grep -rE "@ts-(ignore|expect-error)" --include='*.ts' --include='*.tsx' backend/src frontend/src shared/src 2>/dev/null || true`,
    { cwd: root },
  );
  const lines = output.trim().split("\n").filter(Boolean);
  return { tsSuppressors: lines.length };
}

// Backend tests live in backend/test/; frontend and shared co-locate theirs
// next to the source, so restrict those roots to *.test.* or the grep would
// count `it(`/`expect(` occurrences in production code too.
function collectTestAssertions(root = ROOT) {
  const output = run(
    `grep -rE "\\b(it|test|expect)\\s*\\(" backend/test --include='*.ts' --include='*.tsx' 2>/dev/null; ` +
      `grep -rE "\\b(it|test|expect)\\s*\\(" frontend/src shared/src --include='*.test.ts' --include='*.test.tsx' 2>/dev/null; true`,
    { cwd: root },
  );
  const lines = output.trim().split("\n").filter(Boolean);
  return { testAssertions: lines.length };
}

// ─── Doc contract ─────────────────────────────────────────────────────────────
//
// Enforces the company rule "Every `.md` is a critical file" (standards.md,
// RATIFIED 2026-07-21): exactly two documents may claim authority over a product
// — the PRD and the intake ledger (`IN-NN`). That rule is otherwise enforced only
// by the `ask` permission tier, which stops an agent writing a doc while the human
// is watching. This is the half that holds when nobody is.
//
// Every file under `docs/` must EITHER be on the allowlist below OR open with a
// status banner marking it non-normative. Agents write docs far faster than anyone
// can audit them; the census that triggered the rule found 1,179 authored `.md`
// files across the workspace, one product carrying 160, with nothing to
// distinguish a dated session snapshot from a live spec — so the next agent greps,
// finds a photograph, and reads it as law.
//
// PER-PROJECT SETUP: fill DOC_ALLOWLIST with this project's PRD, its shipped
// artifacts (legal pages compiled into an image), its operational runbook, and any
// bilateral contract that is not unilaterally deletable. Nothing else. Then
// baseline the metric at **0** — a floor, not a ratchet, so a new unstamped doc
// fails on its first commit instead of creeping the allowance upward.

const DOC_ALLOWLIST = new Set([
  // "docs/prd/<product>.md",       // THE PRD — the sole normative product doc
  // "docs/RUNBOOK.md",             // authoritative about the MACHINE, not the product
  // "docs/legal/privacy-policy.md" // shipped artifact — source, not docs
]);

const DOC_BANNER =
  /^>\s*⚠️\s*\*\*(REGISTRO HISTÓRICO|PESQUISA \/ EXPLORAÇÃO|RASCUNHO NÃO RATIFICADO)/;

// Populated by collectDocContract so a regression can name the offending files.
let docContractOffenders = [];

function collectDocContract(root = ROOT) {
  const output = run(`find docs -name '*.md' 2>/dev/null || true`, { cwd: root });
  const files = output.trim().split("\n").filter(Boolean);
  const offenders = files.filter((f) => {
    const rel = f.replace(/^\.\//, "");
    if (DOC_ALLOWLIST.has(rel)) return false;
    try {
      const [first] = readFileSync(path.join(root, rel), "utf8").split("\n", 1);
      return !DOC_BANNER.test(first);
    } catch {
      return true;
    }
  });
  if (root === ROOT) docContractOffenders = offenders;
  return { docContractViolations: offenders.length };
}

function collectCoverage() {
  const readPct = (p) => {
    try {
      const total = JSON.parse(readFileSync(p, "utf8")).total;
      return {
        statements: total.statements.pct,
        branches: total.branches.pct,
        functions: total.functions.pct,
        lines: total.lines.pct,
      };
    } catch {
      return null;
    }
  };
  const zero = { statements: 0, branches: 0, functions: 0, lines: 0 };
  const be = readPct(path.join(ROOT, "backend/coverage/coverage-summary.json")) ?? zero;
  const fe = readPct(path.join(ROOT, "frontend/coverage/coverage-summary.json")) ?? zero;
  return {
    coverageBackendStatements: be.statements,
    coverageBackendBranches: be.branches,
    coverageBackendFunctions: be.functions,
    coverageBackendLines: be.lines,
    coverageFrontendStatements: fe.statements,
    coverageFrontendBranches: fe.branches,
    coverageFrontendFunctions: fe.functions,
    coverageFrontendLines: fe.lines,
  };
}

// ─── Collect all metrics ──────────────────────────────────────────────────────

console.log("Collecting metrics...\n");

const metrics = {
  ...collectFileSizes(),
  ...collectComplexity(),
  ...collectKnip(),
  ...collectJscpd(),
  ...collectAnyCasts(),
  ...collectTsSuppressors(),
  ...collectTestAssertions(),
  ...collectDocContract(),
  ...collectCoverage(),
};

// ─── Read/write baseline ──────────────────────────────────────────────────────

if (UPDATE_BASELINE) {
  writeFileSync(BASELINE_FILE, JSON.stringify(metrics, null, 2) + "\n");
  console.log("Baseline updated. New floor:");
  for (const [k, v] of Object.entries(metrics)) {
    console.log(`  ${k}: ${v}`);
  }
  process.exit(0);
}

const activeBaselineFile = BASELINE_OVERRIDE ?? BASELINE_FILE;

if (!existsSync(activeBaselineFile)) {
  console.error(
    "No baseline found. Run `pnpm quality-gate:update` first to snapshot the current state.",
  );
  process.exit(1);
}

const baseline = JSON.parse(readFileSync(activeBaselineFile, "utf8"));

// ─── Compare ─────────────────────────────────────────────────────────────────

// Metrics where LOWER is better (regressions = current > baseline)
const lowerIsBetter = [
  "maxFileLinesBackend",
  "maxFileLinesFrontend",
  "filesOver500",
  "complexityViolations",
  "unusedExports",
  "unusedTypes",
  "unusedFiles",
  "unusedDependencies",
  "jscpdClones",
  "anyCasts",
  "tsSuppressors",
  "docContractViolations",
];

// Metrics where HIGHER is better (regressions = current < baseline)
const higherIsBetter = [
  "testAssertions",
  "coverageBackendStatements",
  "coverageBackendBranches",
  "coverageBackendFunctions",
  "coverageBackendLines",
  "coverageFrontendStatements",
  "coverageFrontendBranches",
  "coverageFrontendFunctions",
  "coverageFrontendLines",
];

// Per-metric tolerance (absolute) for noise — drift below this counts as SAME.
// V8 coverage % can drift fractionally across runs from unrelated code; 0.5pp
// is large enough to absorb that and small enough to catch real regressions.
const tolerance = {
  coverageBackendStatements: 0.5,
  coverageBackendBranches: 0.5,
  coverageBackendFunctions: 0.5,
  coverageBackendLines: 0.5,
  coverageFrontendStatements: 0.5,
  coverageFrontendBranches: 0.5,
  coverageFrontendFunctions: 0.5,
  coverageFrontendLines: 0.5,
};

const metricLabels = {
  maxFileLinesBackend: "max lines (backend)",
  maxFileLinesFrontend: "max lines (frontend)",
  filesOver500: "files >500 lines",
  complexityViolations: "complexity violations",
  unusedExports: "unused exports",
  unusedTypes: "unused types",
  unusedFiles: "unused files",
  unusedDependencies: "unused deps",
  jscpdClones: "jscpd clones",
  jscpdPercentage: "jscpd %",
  anyCasts: "any casts",
  tsSuppressors: "ts-suppress",
  docContractViolations: "doc-contract violations",
  testAssertions: "test assertions",
  coverageBackendStatements: "coverage backend stmts %",
  coverageBackendBranches: "coverage backend branches %",
  coverageBackendFunctions: "coverage backend funcs %",
  coverageBackendLines: "coverage backend lines %",
  coverageFrontendStatements: "coverage frontend stmts %",
  coverageFrontendBranches: "coverage frontend branches %",
  coverageFrontendFunctions: "coverage frontend funcs %",
  coverageFrontendLines: "coverage frontend lines %",
};

// ─── Drift attribution ─────────────────────────────────────────────────────────
//
// Coverage metrics can't be re-derived from a tree without a (costly) test run, so
// they are excluded from attribution and always block as before. Every other metric
// is STATIC — re-measurable from a checked-out tree with the same tooling — so when
// one regresses we can ask: was the regression already present at the merge-base?
// If so it is pre-existing drift (warn, don't block); if introduced here, block.

const COVERAGE_KEYS = new Set([
  "coverageBackendStatements",
  "coverageBackendBranches",
  "coverageBackendFunctions",
  "coverageBackendLines",
  "coverageFrontendStatements",
  "coverageFrontendBranches",
  "coverageFrontendFunctions",
  "coverageFrontendLines",
]);

const isStatic = (key) => !COVERAGE_KEYS.has(key);

// Which collector re-derives each static metric (run once per collector, memoized).
const METRIC_COLLECTORS = {
  maxFileLinesBackend: collectFileSizes,
  maxFileLinesFrontend: collectFileSizes,
  filesOver500: collectFileSizes,
  complexityViolations: collectComplexity,
  unusedExports: collectKnip,
  unusedTypes: collectKnip,
  unusedFiles: collectKnip,
  unusedDependencies: collectKnip,
  jscpdClones: collectJscpd,
  anyCasts: collectAnyCasts,
  tsSuppressors: collectTsSuppressors,
  testAssertions: collectTestAssertions,
  docContractViolations: collectDocContract,
};

// Resolve the tree to attribute against: merge-base with origin/main, else with
// main. Returns "" when no base is resolvable (→ attribution skipped, fail-closed).
function resolveBaseRef() {
  for (const ref of ["origin/main", "main"]) {
    const sha = run(`git merge-base HEAD ${ref}`, { cwd: ROOT }).trim();
    if (sha) return sha;
  }
  return "";
}

// Classify a regressed static metric given its value re-measured at the base tree.
// undefined base value (collector failed) → "unknown" (blocks, fail-closed).
function classifyDrift(key, baseVal) {
  if (baseVal === undefined || baseVal === null) return "unknown";
  const baselineRaw = baseline[key];
  const baselineNum = typeof baselineRaw === "string" ? parseFloat(baselineRaw) : baselineRaw;
  const baseNum = typeof baseVal === "string" ? parseFloat(baseVal) : baseVal;
  const currentRaw = metrics[key];
  const currentNum = typeof currentRaw === "string" ? parseFloat(currentRaw) : currentRaw;
  if (Number.isNaN(baseNum) || Number.isNaN(baselineNum) || Number.isNaN(currentNum)) {
    return "unknown";
  }
  const tol = tolerance[key] ?? 0;
  if (lowerIsBetter.includes(key)) {
    // Regression = current > baseline (more is worse).
    const presentAtBase = baseNum > baselineNum + tol;
    if (!presentAtBase) return "introduced";
    return currentNum > baseNum + tol ? "partial" : "preExisting";
  }
  // higher-is-better: regression = current < baseline (less is worse).
  const presentAtBase = baseNum < baselineNum - tol;
  if (!presentAtBase) return "introduced";
  return currentNum < baseNum - tol ? "partial" : "preExisting";
}

// Re-measure the regressed static metrics against `baseRef` in a throwaway
// worktree. Returns { metric: classification } for each key.
function attributeDrift(regressedStaticKeys, baseRef) {
  const results = {};
  const tmp = mkdtempSync(path.join(os.tmpdir(), "qg-base-"));
  const worktree = path.join(tmp, "tree");
  let added = false;
  try {
    run(`git worktree add --detach ${JSON.stringify(worktree)} ${baseRef}`, { cwd: ROOT });
    if (!existsSync(path.join(worktree, "package.json"))) {
      // Checkout failed — cannot attribute anything; fail-closed.
      for (const key of regressedStaticKeys) results[key] = "unknown";
      return results;
    }
    added = true;

    // `pnpm exec` collectors (biome/knip/jscpd) resolve binaries from
    // node_modules, which a bare worktree lacks. Symlink the installed trees
    // so those collectors run against base source with current tooling. Pure
    // grep/find collectors need none of this.
    for (const rel of ["node_modules", "backend/node_modules", "frontend/node_modules"]) {
      try {
        const src = path.join(ROOT, rel);
        const dst = path.join(worktree, rel);
        if (existsSync(src) && !existsSync(dst)) {
          ensureDir(path.dirname(dst));
          symlinkSync(src, dst, "dir");
        }
      } catch {
        // best effort — a failed symlink just leaves its collectors "unknown"
      }
    }

    const cache = new Map();
    const measure = (fn) => {
      if (!cache.has(fn)) {
        try {
          cache.set(fn, fn(worktree));
        } catch {
          cache.set(fn, null);
        }
      }
      return cache.get(fn);
    };

    for (const key of regressedStaticKeys) {
      const fn = METRIC_COLLECTORS[key];
      const measured = fn ? measure(fn) : null;
      const baseVal = measured == null ? undefined : measured[key];
      results[key] = classifyDrift(key, baseVal);
    }
    return results;
  } catch {
    for (const key of regressedStaticKeys) {
      if (results[key] === undefined) results[key] = "unknown";
    }
    return results;
  } finally {
    if (added) run(`git worktree remove --force ${JSON.stringify(worktree)}`, { cwd: ROOT });
    run("git worktree prune", { cwd: ROOT });
    try {
      rmSync(tmp, { recursive: true, force: true });
    } catch {
      // ignore
    }
  }
}

let regressions = 0;
let improvements = 0;

const deltas = {};

for (const key of [...lowerIsBetter, ...higherIsBetter]) {
  const current = metrics[key];
  const base = baseline[key] ?? current;
  const baseNum = typeof base === "string" ? parseFloat(base) : base;
  const currentNum = typeof current === "string" ? parseFloat(current) : current;
  const diff = currentNum - baseNum;
  const tol = tolerance[key] ?? 0;
  const isLower = lowerIsBetter.includes(key);

  let status;
  if (Math.abs(diff) <= tol) {
    status = "SAME";
  } else if (isLower) {
    if (diff > 0) {
      status = "REGRESSION";
      regressions++;
    } else {
      status = "IMPROVED";
      improvements++;
    }
  } else {
    if (diff < 0) {
      status = "REGRESSION";
      regressions++;
    } else {
      status = "IMPROVED";
      improvements++;
    }
  }

  deltas[key] = { baseline: base, current, diff, status };
}

// ─── Attribute static regressions (only when there are any) ─────────────────────

const regressedKeys = Object.keys(deltas).filter((k) => deltas[k].status === "REGRESSION");
const staticRegressedKeys = regressedKeys.filter(isStatic);

let attribution = null;

if (!NO_ATTRIBUTION && staticRegressedKeys.length > 0) {
  const baseRef = resolveBaseRef();
  const head = run("git rev-parse HEAD", { cwd: ROOT }).trim();
  const clean = run("git status --porcelain", { cwd: ROOT }).trim() === "";

  // Nothing to attribute when we're on the base commit with a clean tree — the
  // regression is fully committed here; block as before.
  const skip = !baseRef || (baseRef === head && clean);

  if (!skip) {
    console.log(`\nAttributing static regressions against ${baseRef.slice(0, 12)}...`);
    attribution = { ref: baseRef, results: attributeDrift(staticRegressedKeys, baseRef) };
  }
}

// A regression BLOCKS unless it is a static metric proven pre-existing at base.
const classificationOf = (key) => attribution?.results?.[key];
const isBlocking = (key) =>
  deltas[key].status === "REGRESSION" &&
  !(isStatic(key) && classificationOf(key) === "preExisting");

const blocking = regressedKeys.filter(isBlocking).length;
const drifted = regressedKeys.filter(
  (k) => isStatic(k) && classificationOf(k) === "preExisting",
).length;

// ─── Print report ─────────────────────────────────────────────────────────────

const overall = blocking > 0 ? "FAIL" : "PASS";

const summary =
  blocking === 0
    ? drifted > 0
      ? `${drifted} pre-existing drift, not blocking`
      : `${improvements} improvement${improvements !== 1 ? "s" : ""}`
    : `${blocking} blocking regression${blocking !== 1 ? "s" : ""}`;

console.log(`Quality gate: ${overall} (${summary})`);

for (const [key, { baseline: base, current, diff, status }] of Object.entries(deltas)) {
  const drift = isStatic(key) && classificationOf(key) === "preExisting";
  const icon = status === "IMPROVED" ? "✓" : status === "REGRESSION" ? (drift ? "~" : "✗") : "=";
  const label = metricLabels[key] ?? key;
  const diffStr =
    diff === 0
      ? ""
      : diff > 0
        ? ` (+${diff}${status === "REGRESSION" ? (drift ? ", drift" : ", REGRESSION") : ", increased"})`
        : ` (${diff}${status === "IMPROVED" ? ", improved" : drift ? ", drift" : ", decreased"})`;
  const padLabel = label.padEnd(28);
  console.log(`  ${icon} ${padLabel} ${base} → ${current}${diffStr}`);
  // A count alone is unactionable for the doc contract: the whole failure mode is
  // "somebody added a doc nobody will notice", so name the files or the gate just
  // says a number went up.
  if (key === "docContractViolations" && status === "REGRESSION" && docContractOffenders.length) {
    for (const f of docContractOffenders) {
      console.log(`      ↳ ${f} — allowlist it in scripts/quality-gate.mjs, stamp it non-normative, or delete it`);
    }
  }
}

// ─── Pre-existing drift warning block ───────────────────────────────────────────

if (attribution) {
  const driftKeys = staticRegressedKeys.filter((k) => attribution.results[k] === "preExisting");
  const partialKeys = staticRegressedKeys.filter((k) => attribution.results[k] === "partial");

  if (driftKeys.length > 0) {
    console.log(
      `\n⚠  Pre-existing drift (present at ${attribution.ref.slice(0, 12)}) — not caused by this change:`,
    );
    for (const k of driftKeys) console.log(`     - ${metricLabels[k] ?? k}`);
    console.log(
      "   To reconcile: get owner approval, then `pnpm quality-gate:update`.\n" +
        "   These do NOT block this run.",
    );
  }
  if (partialKeys.length > 0) {
    console.log(
      `\n✗  Worsened beyond a pre-existing baseline (the delta past ${attribution.ref.slice(0, 12)} is yours):`,
    );
    for (const k of partialKeys) console.log(`     - ${metricLabels[k] ?? k}`);
  }
}

// ─── Write report.json ────────────────────────────────────────────────────────

ensureDir(REPORT_DIR);
writeFileSync(
  REPORT_FILE,
  JSON.stringify(
    {
      timestamp: new Date().toISOString(),
      overall,
      regressions,
      improvements,
      metrics,
      baseline,
      deltas,
      ...(attribution ? { attribution } : {}),
    },
    null,
    2,
  ) + "\n",
);

// ─── Exit code ────────────────────────────────────────────────────────────────

process.exit(blocking > 0 ? 1 : 0);
