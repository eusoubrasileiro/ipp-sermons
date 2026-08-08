import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { createShellLoader } from "../../src/lib/seo/shell.ts";

/**
 * The built SPA shell is read from disk once and held.
 *
 * It is baked into the image and changes only when a release replaces the
 * container, so re-reading it on every crawler request would be 6 KB of I/O for
 * nothing. A missing shell is not an error: the dev backend runs without the
 * frontend having been built, and the site must simply behave as it does today.
 */

const dir = join(tmpdir(), `ipp-shell-test-${process.pid}`);

beforeAll(async () => {
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "index.html"), "<html>primeiro</html>", "utf8");
});

describe("createShellLoader", () => {
  it("reads the shell and then serves it from memory", async () => {
    const load = createShellLoader(dir);

    expect(await load()).toBe("<html>primeiro</html>");
    await writeFile(join(dir, "index.html"), "<html>segundo</html>", "utf8");
    expect(await load()).toBe("<html>primeiro</html>");
  });

  it("returns null when there is no build, rather than throwing", async () => {
    expect(await createShellLoader(join(dir, "nao-existe"))()).toBe(null);
  });

  it("does not cache the absence, so a shell written later is picked up", async () => {
    const late = join(dir, "tarde");
    const load = createShellLoader(late);

    expect(await load()).toBe(null);
    await mkdir(late, { recursive: true });
    await writeFile(join(late, "index.html"), "<html>tarde</html>", "utf8");
    expect(await load()).toBe("<html>tarde</html>");
  });
});
