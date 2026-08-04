# Agent contract — read before any tool call

## Task contract — `.claude/PLAN.md`

Your task for this dispatch is described in `.claude/PLAN.md`.

**First action — always:** read `.claude/PLAN.md`. If the file is missing or empty, abort immediately and report "no plan in worktree — dispatch bypassed the gate". Do not improvise scope from external context. The plan file is the contract; if it doesn't cover something, ask the leader before doing it.

## How to work

You are running inside a dispatched worktree. Hard rules:

1. **STAY HERE.** Every command runs from `${WORKTREE_ROOT}`. If `git rev-parse --show-toplevel` doesn't return `${WORKTREE_ROOT}`, STOP and report — you're about to corrupt the main worktree.

2. **Never bypass hooks or signing.** No `--no-verify`, no disabled signing. If a hook fails, fix the root cause.

3. **Source `.agent-env` for any DB or server command.** It overrides `DATABASE_URL` (your own per-slug database) and the port vars for this worktree; for ad-hoc commands prefix with `set -a && . .agent-env && set +a`. Hit the backend at `127.0.0.1:${BACKEND_PORT}`. The shared dev database is not yours — indexing into it destroys another agent's corpus.

4. **Schema changes go through `pnpm db:push`, never a bare `prisma db push`.** The bare command drops the generated `fts` column and takes the GIN index and `hybrid_search()` with it. See `CLAUDE.md` § "Traps worth knowing".

5. **Branch is `agent/${AGENT_SLUG}`. Don't switch, don't pull from main mid-task** (ask the leader for a rebase). Push to `origin/agent/${AGENT_SLUG}` and end with `pnpm pr:create --base main`. You never merge.

6. **Critical files require human approval** — see `CLAUDE.md` § "Critical files". If your task seems to need editing one, STOP and ask the leader.

7. **Pre-push is heavy and ends in an LLM reviewer.** Reject justifications and concerns print to stderr — read them and fix the root cause. Max 3 attempts, as a ladder: (1) mechanical fix for lint/format/import failures, (2) one scoped model-fix, (3) escalate to the leader. After the 3rd reject: STOP, do not push, do not open a PR. Report the final reject reason, `git diff --stat main...HEAD`, and a one-line note per attempt explaining what you changed.

## When you finish

```bash
git add <files>
git commit -m "feat(scope): ..."   # conventional commits enforced by commit-msg
git push -u origin agent/${AGENT_SLUG}
pnpm pr:create --base main --head agent/${AGENT_SLUG} --title "..." --body "..."
```

Report back with the PR URL. The leader takes it from there.
