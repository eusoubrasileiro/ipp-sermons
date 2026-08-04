/**
 * lint-staged runs FIRST in .husky/pre-commit, and the hook then runs
 * `pnpm typecheck` and `pnpm test:coverage` itself. So everything listed here
 * must be something the hook does NOT already do, or it runs twice on every
 * commit for no added signal. That leaves lint.
 *
 * The `() => [...]` form drops the staged-file list deliberately: `biome check`
 * is fast enough repo-wide, and a whole-repo check cannot be fooled by a change
 * whose fallout lands in a file that was not staged.
 */
export default {
  "*.{ts,tsx,js}": () => ["pnpm lint"],
};
