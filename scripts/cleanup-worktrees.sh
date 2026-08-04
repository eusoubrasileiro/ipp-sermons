#!/usr/bin/env bash
# cleanup-worktrees.sh (see standards/standards.md §7)
#
# Tear down a dispatched worktree.
#
# Usage:
#   pnpm dispatch:cleanup --slug <slug> [--force]
#
# Steps:
#   1. git worktree remove .claude/worktrees/<slug>  (--force when dirty)
#   2. DROP DATABASE "<DB_NAME_PREFIX>-<slug>"        (idempotent)
#   3. git branch -D agent/<slug>                     (only if merged, or --force)
#   4. git worktree prune
#
# PG_USER / PG_PASS / PG_PORT / DB_NAME_PREFIX must stay in sync with
# dispatch-worktree.sh — a mismatch silently leaves the per-slug DB behind.
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  pnpm dispatch:cleanup --slug <slug> [--force]   # remove one slug
EOF
  exit 2
}

TARGET_SLUG=""
FORCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --slug)
      [ $# -ge 2 ] || usage
      TARGET_SLUG="$2"; shift 2 ;;
    --force)
      FORCE="force"; shift ;;
    -h|--help)
      usage ;;
    *)
      echo "Unknown arg: $1" >&2; usage ;;
  esac
done
[ -n "$TARGET_SLUG" ] || usage

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMMON_DIR="$(git -C "$SCRIPT_DIR" rev-parse --git-common-dir)"
case "$COMMON_DIR" in
  /*) COMMON_DIR_ABS="$COMMON_DIR" ;;
  *)  COMMON_DIR_ABS="$(cd "$SCRIPT_DIR" && cd "$COMMON_DIR" && pwd)" ;;
esac
PARENT_ROOT="$(cd "$COMMON_DIR_ABS/.." && pwd)"
cd "$PARENT_ROOT"

PG_USER="${PG_USER:-ipp}"
PG_PASS="${PG_PASS:-ipp}"
PG_PORT="${PG_PORT:-5439}"
DB_NAME_PREFIX="${DB_NAME_PREFIX:-ipp-agent}"

PGPASSWORD="$PG_PASS"
export PGPASSWORD
PSQL=(psql -U "$PG_USER" -h 127.0.0.1 -p "$PG_PORT" -d postgres -v ON_ERROR_STOP=1)

drop_db_if_exists() {
  local db="$1"
  if ! nc -z 127.0.0.1 "$PG_PORT" 2>/dev/null; then
    echo "[cleanup] Postgres not reachable — skipping DROP DATABASE \"$db\""
    return 0
  fi
  # Terminate stragglers, then drop.
  "${PSQL[@]}" -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$db' AND pid<>pg_backend_pid()" >/dev/null 2>&1 || true
  "${PSQL[@]}" -c "DROP DATABASE IF EXISTS \"$db\"" || true
}

cleanup_one() {
  local slug="$1"
  local force="${2:-}"
  local wt=".claude/worktrees/$slug"
  local branch="agent/$slug"
  local db="$DB_NAME_PREFIX-$slug"

  echo "[cleanup] slug=$slug branch=$branch wt=$wt db=$db"

  # 1. Remove the worktree.
  if git -C "$PARENT_ROOT" worktree list --porcelain | awk '$1=="worktree"{print $2}' | grep -Fxq "$PARENT_ROOT/$wt"; then
    if [ "$force" = "force" ]; then
      git -C "$PARENT_ROOT" worktree remove --force "$wt" || true
    else
      git -C "$PARENT_ROOT" worktree remove "$wt" || {
        echo "[cleanup] worktree $wt is dirty — pass --all to force, or commit/stash first." >&2
        return 1
      }
    fi
  else
    # Not a registered worktree; remove the directory if it's a stale dir.
    if [ -e "$PARENT_ROOT/$wt" ]; then
      rm -rf "$PARENT_ROOT/$wt"
    fi
  fi

  # 2. Drop DB.
  drop_db_if_exists "$db"

  # 3. Delete branch.
  if git -C "$PARENT_ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
    if [ "$force" = "force" ]; then
      git -C "$PARENT_ROOT" branch -D "$branch" || true
    else
      # Only delete if merged into main. Robust to branch being checked out.
      if git -C "$PARENT_ROOT" merge-base --is-ancestor "$branch" main 2>/dev/null; then
        git -C "$PARENT_ROOT" branch -D "$branch"
      else
        echo "[cleanup] keeping unmerged branch $branch (use --force to delete anyway)."
      fi
    fi
  fi

  # 4. Prune git metadata.
  git -C "$PARENT_ROOT" worktree prune
}

if ! printf '%s' "$TARGET_SLUG" | grep -Eq '^[a-z0-9-]{1,40}$'; then
  echo "Error: slug must match [a-z0-9-]{1,40}" >&2
  exit 1
fi
cleanup_one "$TARGET_SLUG" "$FORCE"

echo "[cleanup] done."
