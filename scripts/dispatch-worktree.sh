#!/usr/bin/env bash
# dispatch-worktree.sh (see standards/standards.md §7)
#
# Materialise an "agent-ready" git worktree for a sub-agent: isolated branch,
# hash-derived ports, per-slug Postgres DB, symlinked env files, stamped
# .claude/AGENT.md contract.
#
# Usage:
#   pnpm dispatch <slug> [--from <base-branch>] [--no-install]
#
# Emits a KEY=value block on stdout (last lines):
#   WORKTREE_ROOT=...
#   AGENT_PORT_BACKEND=...
#   AGENT_PORT_FRONTEND=...
#   AGENT_DB_NAME=...
#   AGENT_SLUG=...
#
# Adapted for this repo: creds match the local `pnpm db:up` compose service
# (docker-compose.yml), `.env` is the only required env file (there is no
# .env.test — unit tests hit neither DB nor network), and the post-install step
# builds @ipp/shared and generates the Prisma client because the pre-commit
# typecheck cannot run without either.
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: pnpm dispatch <slug> [--from <base-branch>] [--no-install]

Creates a git worktree under .claude/worktrees/<slug>/ on branch agent/<slug>,
symlinks .env from the parent, allocates a unique
port-and-DB triplet derived from the slug hash, and writes a per-agent
.agent-env override file inside the worktree.

Slug must match [a-z0-9-]{1,40}.
EOF
  exit 2
}

# ── parse args ──────────────────────────────────────────────────────────────
SLUG=""
BASE_BRANCH="main"
DO_INSTALL=1
while [ $# -gt 0 ]; do
  case "$1" in
    --from)
      [ $# -ge 2 ] || usage
      BASE_BRANCH="$2"; shift 2 ;;
    --no-install)
      DO_INSTALL=0; shift ;;
    -h|--help)
      usage ;;
    --*)
      echo "Unknown flag: $1" >&2; usage ;;
    *)
      [ -z "$SLUG" ] || { echo "Multiple slugs: $SLUG and $1" >&2; usage; }
      SLUG="$1"; shift ;;
  esac
done

[ -n "$SLUG" ] || usage

if ! printf '%s' "$SLUG" | grep -Eq '^[a-z0-9-]{1,40}$'; then
  echo "Error: slug must match [a-z0-9-]{1,40} (got: $SLUG)" >&2
  exit 1
fi

# ── resolve parent (main-worktree) tree ─────────────────────────────────────
# Always resolve PARENT_ROOT to the main worktree, regardless of where the
# script was invoked from. `git rev-parse --git-common-dir` points at the main
# worktree's .git directory (or the bare repo); its parent is the main worktree.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMMON_DIR="$(git -C "$SCRIPT_DIR" rev-parse --git-common-dir)"
case "$COMMON_DIR" in
  /*) COMMON_DIR_ABS="$COMMON_DIR" ;;
  *)  COMMON_DIR_ABS="$(cd "$SCRIPT_DIR" && cd "$COMMON_DIR" && pwd)" ;;
esac
PARENT_ROOT="$(cd "$COMMON_DIR_ABS/.." && pwd)"

cd "$PARENT_ROOT"

# ── sanity gate: env file present ───────────────────────────────────────────
# `.env` carries OPENROUTER_API_KEY and DATABASE_URL. Unit tests need neither,
# but `pnpm index` and `pnpm eval` do, and an agent that discovers this halfway
# through a task has already wasted the dispatch.
if [ ! -e ".env" ]; then
  echo "Error: no .env in $PARENT_ROOT (copy .env.example and fill it in)." >&2
  exit 1
fi

# ── sanity gate: plan file present ──────────────────────────────────────────
# The leader MUST write `.claude/plans/<slug>.md` BEFORE invoking dispatch.
# That file is the sub-agent's task contract (copied into the worktree as
# `.claude/PLAN.md`) and the "why this PR exists" source for `pnpm pr:create`.
# Hard fail: no worktree, no DB, no port allocation until the plan exists.
#
# IMPORTANT: `.claude/plans/` MUST be gitignored in the project's .gitignore
# (same tier as `.claude/worktrees/`). Otherwise, when concurrent leaders work
# side-by-side in the shared parent worktree, each leader's untracked
# `<slug>.md` dirties the tree for the OTHER leaders and blocks their dispatch
# at the parent-clean gate below. Gitignored = invisible to git status but
# still present on disk, so this plan-exists check keeps working.
PLAN_SRC="$PARENT_ROOT/.claude/plans/$SLUG.md"
if [ ! -s "$PLAN_SRC" ]; then
  echo "Error: no plan found at .claude/plans/$SLUG.md (or file is empty)." >&2
  echo "       Write the sub-agent's task plan first, then re-run dispatch." >&2
  exit 1
fi

# ── sanity gate: parent tree clean enough ───────────────────────────────────
# Allow *.log, frontend/playwright-report/, .quality-gate/ — those are
# regeneratable artefacts the leader's WIP shouldn't be considered "dirty" for.
DIRTY="$(git -C "$PARENT_ROOT" status --porcelain |
  grep -Ev '(\.log$|^.. \.quality-gate/|^.. coverage/)' || true)"
if [ -n "$DIRTY" ]; then
  echo "Error: parent worktree has uncommitted changes that aren't ignorable artefacts." >&2
  echo "       Dispatch refuses to drag your WIP into the agent worktree." >&2
  echo >&2
  git -C "$PARENT_ROOT" status >&2
  exit 1
fi

# ── Postgres creds ──────────────────────────────────────────────────────────
# Match the `db` service in docker-compose.yml (`pnpm db:up`). 5439 is the host
# port there — deliberately not 5432, so a dispatch never lands on a system
# Postgres by accident.
PG_USER="${PG_USER:-ipp}"
PG_PASS="${PG_PASS:-ipp}"
PG_PORT="${PG_PORT:-5439}"
DB_NAME_PREFIX="${DB_NAME_PREFIX:-ipp-agent}"

# ── ensure local Postgres pod is up ─────────────────────────────────────────
if ! nc -z 127.0.0.1 "$PG_PORT" 2>/dev/null; then
  echo "[dispatch] Postgres down on 127.0.0.1:$PG_PORT — starting it..."
  (cd "$PARENT_ROOT" && docker compose up -d db) || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    nc -z 127.0.0.1 "$PG_PORT" 2>/dev/null && break
    sleep 1
  done
fi

if ! nc -z 127.0.0.1 "$PG_PORT" 2>/dev/null; then
  echo "Error: Postgres pod not reachable on 127.0.0.1:$PG_PORT." >&2
  echo "       Start it (e.g. \`pnpm db:up\`) and re-run dispatch." >&2
  exit 1
fi

# ── allocate slot from sha1(slug) ───────────────────────────────────────────
HEX_SLOT="$(printf '%s' "$SLUG" | sha1sum | cut -c1-2)"
SLOT_DEC=$((16#${HEX_SLOT}))
BACKEND_PORT=$((3100 + SLOT_DEC * 4))
FRONTEND_PORT=$((BACKEND_PORT + 1))
MSW_PORT=$((BACKEND_PORT + 2))
PLAYWRIGHT_HTML_PORT=$((BACKEND_PORT + 3))

WORKTREES_DIR="$PARENT_ROOT/.claude/worktrees"
WORKTREE_ROOT="$WORKTREES_DIR/$SLUG"

# ── port collision check ────────────────────────────────────────────────────
# If a port is bound, check whether it's bound by THIS slug's existing worktree
# (idempotent re-dispatch is OK) or by something else (refuse).
check_port() {
  local port="$1"
  if ! lsof -ti:"$port" >/dev/null 2>&1; then
    return 0
  fi
  # Port is bound. Refuse — we never know whether it's our own resumed agent
  # or a stranger; the safe move is to ask the operator to free it.
  local pids
  pids="$(lsof -ti:"$port" || true)"
  echo "Error: port $port (slot $SLOT_DEC) already in use by pid(s): $pids" >&2
  echo "       Free the port or pick a different slug." >&2
  echo "       (Re-dispatching the SAME slug while its agent is running is not supported.)" >&2
  exit 1
}
check_port "$BACKEND_PORT"
check_port "$FRONTEND_PORT"
check_port "$MSW_PORT"
check_port "$PLAYWRIGHT_HTML_PORT"

# ── create / reuse worktree ─────────────────────────────────────────────────
mkdir -p "$WORKTREES_DIR"
BRANCH="agent/$SLUG"

EXISTING_WT="$(git -C "$PARENT_ROOT" worktree list --porcelain |
  awk -v p="$WORKTREE_ROOT" '$1=="worktree" && $2==p {found=1} END{print (found?"yes":"")}')"

if [ -n "$EXISTING_WT" ]; then
  echo "[dispatch] reusing existing worktree at $WORKTREE_ROOT"
  # Verify branch matches expectation.
  CUR_BRANCH="$(git -C "$WORKTREE_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")"
  if [ "$CUR_BRANCH" != "$BRANCH" ]; then
    echo "Error: worktree at $WORKTREE_ROOT is on branch '$CUR_BRANCH', expected '$BRANCH'." >&2
    echo "       Pick a different slug or clean up via pnpm dispatch:cleanup --slug $SLUG." >&2
    exit 1
  fi
else
  # Create a new worktree.
  if git -C "$PARENT_ROOT" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    # Branch exists already — just check it out into the new path.
    echo "[dispatch] branch $BRANCH already exists; checking out into $WORKTREE_ROOT"
    git -C "$PARENT_ROOT" worktree add "$WORKTREE_ROOT" "$BRANCH"
  else
    echo "[dispatch] creating worktree $WORKTREE_ROOT on new branch $BRANCH (off $BASE_BRANCH)"
    git -C "$PARENT_ROOT" worktree add -b "$BRANCH" "$WORKTREE_ROOT" "$BASE_BRANCH"
  fi
fi

# ── symlink secret env files ────────────────────────────────────────────────
# Symlink whatever the parent has. Skip ones that don't exist (.env.smoke is
# optional; only smoke-vendor tests need it).
for f in .env; do
  if [ -e "$PARENT_ROOT/$f" ]; then
    ln -sfn "$PARENT_ROOT/$f" "$WORKTREE_ROOT/$f"
  fi
done

# ── write per-agent override ────────────────────────────────────────────────
DB_NAME="$DB_NAME_PREFIX-$SLUG"
AGENT_ENV="$WORKTREE_ROOT/.agent-env"
cat > "$AGENT_ENV" <<EOF
# .agent-env — per-agent override, sourced AFTER .env.test by aware scripts.
# Gitignored. Regenerated on every \`pnpm dispatch $SLUG\`.
AGENT_SLUG=$SLUG
WORKTREE_ROOT=$WORKTREE_ROOT
BACKEND_PORT=$BACKEND_PORT
FRONTEND_PORT=$FRONTEND_PORT
MSW_PORT=$MSW_PORT
PLAYWRIGHT_HTML_PORT=$PLAYWRIGHT_HTML_PORT
DATABASE_URL=postgresql://$PG_USER:$PG_PASS@localhost:$PG_PORT/$DB_NAME
EOF

# ── provision per-agent DB ──────────────────────────────────────────────────
# Stock Postgres has no IF NOT EXISTS for CREATE DATABASE; emulate with a
# pg_database lookup.
PGPASSWORD="$PG_PASS"
export PGPASSWORD
PSQL=(psql -U "$PG_USER" -h 127.0.0.1 -p "$PG_PORT" -d postgres -v ON_ERROR_STOP=1)
DB_EXISTS="$("${PSQL[@]}" -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" || true)"
if [ "$DB_EXISTS" = "1" ]; then
  echo "[dispatch] Postgres DB \"$DB_NAME\" already exists; reusing."
else
  echo "[dispatch] creating Postgres DB \"$DB_NAME\"..."
  "${PSQL[@]}" -c "CREATE DATABASE \"$DB_NAME\""
fi

# ── stamp the agent prompt ──────────────────────────────────────────────────
mkdir -p "$WORKTREE_ROOT/.claude"
# Prefer the parent's copy (canonical), fall back to the script's own dir for
# the smoke test where the script lives only in the dispatched worktree.
if [ -f "$PARENT_ROOT/scripts/agent-prompt.md" ]; then
  PROMPT_SRC="$PARENT_ROOT/scripts/agent-prompt.md"
else
  PROMPT_SRC="$SCRIPT_DIR/agent-prompt.md"
fi
PROMPT_DST="$WORKTREE_ROOT/.claude/AGENT.md"
if [ -f "$PROMPT_SRC" ]; then
  # Substitute ${WORKTREE_ROOT} and ${AGENT_SLUG} naively. Keep it as a plain
  # text replacement to avoid surprising the agent with shell expansion.
  sed -e "s|\${WORKTREE_ROOT}|$WORKTREE_ROOT|g" \
      -e "s|\${AGENT_SLUG}|$SLUG|g" \
      "$PROMPT_SRC" > "$PROMPT_DST"
else
  echo "[dispatch] WARNING: $PROMPT_SRC missing; skipping AGENT.md stamp." >&2
fi

# ── copy the leader's plan into the worktree as PLAN.md ─────────────────────
# The agent's task contract. Copied (not symlinked) so that if the
# leader edits the pre-dispatch plan later, the worktree's contract stays
# frozen at dispatch time.
cp "$PLAN_SRC" "$WORKTREE_ROOT/.claude/PLAN.md"

# ── install deps ────────────────────────────────────────────────────────────
# @ipp/shared is consumed as built dist and the Prisma client is generated, not
# committed. Both are gitignored, so a fresh worktree has neither and the
# pre-commit typecheck fails with a confusing "cannot find module @ipp/shared".
if [ "$DO_INSTALL" = "1" ]; then
  echo "[dispatch] running pnpm install --frozen-lockfile in $WORKTREE_ROOT..."
  (cd "$WORKTREE_ROOT" && pnpm install --frozen-lockfile)
  echo "[dispatch] building @ipp/shared..."
  (cd "$WORKTREE_ROOT" && pnpm --filter @ipp/shared build)
  echo "[dispatch] running prisma generate..."
  (cd "$WORKTREE_ROOT" && pnpm --filter @ipp/backend exec prisma generate)
else
  echo "[dispatch] --no-install: skipping pnpm install + post-install steps"
fi

# ── final machine-readable output (LAST lines on stdout) ────────────────────
cat <<EOF

# ── dispatch complete ───────────────────────────────────────────────────────
WORKTREE_ROOT=$WORKTREE_ROOT
AGENT_PORT_BACKEND=$BACKEND_PORT
AGENT_PORT_FRONTEND=$FRONTEND_PORT
AGENT_PORT_MSW=$MSW_PORT
AGENT_PORT_PLAYWRIGHT_HTML=$PLAYWRIGHT_HTML_PORT
AGENT_DB_NAME=$DB_NAME
AGENT_SLUG=$SLUG
EOF
