import { readFile } from "node:fs/promises";
import { join } from "node:path";

/**
 * Reads the built SPA shell, once.
 *
 * It is baked into the image and changes only when a release replaces the
 * container, so re-reading 6 KB on every crawler request would buy nothing.
 *
 * A missing shell returns null rather than throwing, and is deliberately not
 * cached: the dev backend runs before `pnpm --filter @ipp/frontend build` has
 * ever been run, and picking the shell up when it appears is friendlier than
 * requiring a restart. Either way the caller falls through to the static
 * middleware, so the site behaves exactly as it does without any of this.
 */
export type ShellLoader = () => Promise<string | null>;

export function createShellLoader(publicDir: string): ShellLoader {
  let cached: string | null = null;

  return async () => {
    if (cached !== null) return cached;
    try {
      cached = await readFile(join(publicDir, "index.html"), "utf8");
      return cached;
    } catch {
      return null;
    }
  };
}
