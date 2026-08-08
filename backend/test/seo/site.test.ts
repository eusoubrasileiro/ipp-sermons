import { afterEach, describe, expect, it, vi } from "vitest";

/**
 * Where the site thinks it lives.
 *
 * Two constants, but not inert ones: every canonical and Open Graph URL is
 * built on `SITE_URL`, and getting it wrong is the kind of mistake a crawler
 * believes. Telling Google that an internal Docker origin is canonical would
 * de-index the site rather than break it visibly, which is why this is
 * configuration with a default and not `c.req.header("host")`.
 *
 * The module reads `process.env` at import time, so each case re-imports it
 * with the environment it wants.
 */

const ORIGINAL = { ...process.env };

afterEach(() => {
  process.env = { ...ORIGINAL };
  vi.resetModules();
});

async function loadSite(env: Record<string, string | undefined>) {
  vi.resetModules();
  for (const [k, v] of Object.entries(env)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
  return import("../../src/lib/seo/site.ts");
}

describe("SITE_URL", () => {
  it("defaults to production, so a deploy needs no extra configuration", async () => {
    const { SITE_URL } = await loadSite({ PUBLIC_BASE_URL: undefined });
    expect(SITE_URL).toBe("https://ipp-sermons.amiticia.cc");
  });

  it("takes PUBLIC_BASE_URL when the site answers under another name", async () => {
    const { SITE_URL } = await loadSite({ PUBLIC_BASE_URL: "https://sermoes.exemplo.org" });
    expect(SITE_URL).toBe("https://sermoes.exemplo.org");
  });

  it("strips trailing slashes", async () => {
    // Paths are concatenated onto this. A trailing slash yields
    // `https://host//sermao/123`, which is a different URL to a crawler and
    // splits the ranking of every page on the site.
    const { SITE_URL } = await loadSite({ PUBLIC_BASE_URL: "https://exemplo.org///" });
    expect(SITE_URL).toBe("https://exemplo.org");
    expect(`${SITE_URL}/sermao/123`).toBe("https://exemplo.org/sermao/123");
  });
});

describe("PUBLIC_DIR", () => {
  it("defaults to the same ./public that serveStatic uses", async () => {
    // The prerenderer reads index.html from here and the static middleware
    // serves the assets it names. Two different directories would mean a shell
    // pointing at a bundle that is not there.
    const { PUBLIC_DIR } = await loadSite({ PUBLIC_DIR: undefined });
    expect(PUBLIC_DIR).toBe("./public");
  });

  it("is overridable so a test can point at a fixture", async () => {
    const { PUBLIC_DIR } = await loadSite({ PUBLIC_DIR: "/tmp/fixture-public" });
    expect(PUBLIC_DIR).toBe("/tmp/fixture-public");
  });
});
