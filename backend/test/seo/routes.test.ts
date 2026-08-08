import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { PrismaClient } from "@prisma/client";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { createApp } from "../../src/app.ts";
import type { EmbeddingsClient } from "../../src/lib/embeddings.ts";

/**
 * The prerendered pages, driven through the real Hono app.
 *
 * The rule these pin is that prerendering may never make the site worse: a
 * sermon that is not in the database, a transcript that is not on disk, a
 * database that is down and a shell that was never built all have to fall
 * through to the SPA shell exactly as they do today. In these tests "falls
 * through" reads as 404, because nothing is registered after the injector —
 * in `server.ts` the static middleware and the catch-all are.
 */

const base = join(tmpdir(), `ipp-seo-test-${process.pid}`);
const publicDir = join(base, "public");
const FILE = "01-01-2023 - Tito 2.txt";
const TEXT =
  "Como sempre, é um motivo de alegria estar aqui diante dessa querida igreja de Cristo.";

const SHELL = `<!doctype html>
<html lang="pt-BR">
  <head>
    <title>Sermões IPP — Igreja Presbiteriana Peregrinos</title>
    <meta name="description" content="Busca nos sermões." />
    <script type="module" crossorigin src="/assets/index-abc.js"></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>`;

beforeAll(async () => {
  await mkdir(join(base, "transcripts"), { recursive: true });
  await mkdir(publicDir, { recursive: true });
  await writeFile(join(base, "transcripts", FILE), TEXT, "utf8");
  await writeFile(join(publicDir, "index.html"), SHELL, "utf8");
});

const stubEmbeddings = (): EmbeddingsClient => ({
  embed: vi.fn(async (inputs: string[]) => inputs.map(() => [0.1, 0.2, 0.3])),
});

const sermonRow = (over: Record<string, unknown> = {}) => ({
  id: "2123517330",
  title: "01-01-2023 - Tito 2",
  artist: "Reverendo Bruno Melo",
  date: new Date("2023-01-01T00:00:00Z"),
  durationStr: "1:02:39",
  scSuffixUrl: "tito-2",
  spSuffixUrl: "4rOoJ6Egrf8K2IrywzwOMk",
  spotifyAlive: true,
  words: 6424,
  transcriptFile: FILE,
  ...over,
});

/**
 * `facetTree()` issues eight GROUP BYs; each is recognisable by the table it
 * reads, so one stub can answer all of them without pretending to be Postgres.
 */
const facetRows = async (strings: TemplateStringsArray) => {
  const sql = Array.isArray(strings) ? strings.join(" ") : String(strings);
  if (sql.includes("FROM bible_books")) {
    return [{ slug: "tito", nome: "Tito", testamento: "NT", ordem: 56, chapter: 2, total: 1 }];
  }
  if (sql.includes("FROM sermon_scriptures")) return [{ slug: "tito", total: 1 }];
  if (sql.includes("FROM sermons GROUP BY artist")) {
    return [{ artist: "Reverendo Bruno Melo", total: 1 }];
  }
  return [];
};

const stubPrisma = (over: Record<string, unknown> = {}) =>
  ({
    sermon: {
      findUnique: vi.fn(async () => sermonRow()),
      findMany: vi.fn(async () => [sermonRow()]),
      count: vi.fn(async () => 1),
    },
    $queryRaw: vi.fn(facetRows),
    ...over,
  }) as unknown as PrismaClient;

const app = (over: Record<string, unknown> = {}) =>
  createApp({
    prisma: stubPrisma(over),
    embeddings: stubEmbeddings(),
    dataDir: base,
    publicDir,
    siteUrl: "https://exemplo.test",
  });

describe("GET /sermao/:id", () => {
  it("answers a crawler with the sermon's own title, description and text", async () => {
    const res = await app().request("/sermao/2123517330");
    const html = await res.text();

    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/html");
    expect(html).toContain("<title>Tito 2 — Reverendo Bruno Melo</title>");
    expect(html).toContain("motivo de alegria estar aqui diante dessa querida igreja");
    expect(html).toContain(
      '<link rel="canonical" href="https://exemplo.test/sermao/2123517330" />',
    );
  });

  it("still ships the module script, so the SPA hydrates over it", async () => {
    const html = await (await app().request("/sermao/2123517330")).text();
    expect(html).toContain('src="/assets/index-abc.js"');
  });

  it("forbids a browser from holding the page, because it names a hashed bundle", async () => {
    // The document points at /assets/index-<hash>.js and a release changes that
    // hash. A cached copy asks the new container for the old file, gets
    // index.html from the SPA catch-all, and refuses it as a module -- a blank
    // site for the length of the max-age. This shipped once with max-age=3600.
    const res = await app().request("/sermao/2123517330");
    expect(res.headers.get("cache-control")).toBe("public, max-age=0, must-revalidate");
  });

  it("falls through to the SPA shell for a sermon that is not in the database", async () => {
    const prisma = { sermon: { findUnique: vi.fn(async () => null) } };
    expect((await app(prisma).request("/sermao/nao-existe")).status).toBe(404);
  });

  it("falls through rather than 500ing when the database is down", async () => {
    // A prerenderer that turns a working page into an error page is worse than
    // no prerenderer at all.
    const prisma = {
      sermon: {
        findUnique: vi.fn(async () => {
          throw new Error("connect ECONNREFUSED 10.0.0.5:5432");
        }),
      },
    };
    const res = await app(prisma).request("/sermao/2123517330");

    expect(res.status).toBe(404);
    expect(await res.text()).not.toContain("10.0.0.5");
  });

  it("falls through when the frontend has not been built", async () => {
    const res = await createApp({
      prisma: stubPrisma(),
      embeddings: stubEmbeddings(),
      dataDir: base,
      publicDir: join(base, "sem-build"),
      siteUrl: "https://exemplo.test",
    }).request("/sermao/2123517330");

    expect(res.status).toBe(404);
  });

  it("leaves the API routes alone", async () => {
    const res = await app().request("/api/sermons/2123517330/transcript");
    expect(res.headers.get("content-type")).toContain("application/json");
  });
});

describe("GET /sitemap.xml and /robots.txt", () => {
  it("serves the sitemap as XML", async () => {
    const res = await app().request("/sitemap.xml");

    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("xml");
    expect(await res.text()).toContain("<loc>https://exemplo.test/sermao/2123517330</loc>");
  });

  it("serves robots.txt as plain text, pointing at the sitemap", async () => {
    const res = await app().request("/robots.txt");

    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/plain");
    expect(await res.text()).toContain("Sitemap: https://exemplo.test/sitemap.xml");
  });

  it("serves an empty sitemap rather than failing when the database is down", async () => {
    const prisma = {
      sermon: {
        findMany: vi.fn(async () => {
          throw new Error("connect ECONNREFUSED 10.0.0.5:5432");
        }),
      },
      $queryRaw: vi.fn(async () => []),
    };
    const res = await app(prisma).request("/sitemap.xml");

    expect(res.status).toBe(503);
    expect(await res.text()).not.toContain("10.0.0.5");
  });
});

describe("the browse pages", () => {
  it("prerenders a facet leaf with links to the sermons under it", async () => {
    const html = await (await app().request("/biblia/tito")).text();

    expect(html).toContain('href="/sermao/2123517330"');
    expect(html).toContain('<link rel="canonical" href="https://exemplo.test/biblia/tito" />');
  });

  it("prerenders a facet index", async () => {
    const res = await app().request("/pregadores");
    expect(res.status).toBe(200);
    expect(await res.text()).toContain("<title>Pregadores");
  });

  it("prerenders the home page with links into every index", async () => {
    const html = await (await app().request("/")).text();

    expect(html).toContain('href="/biblia"');
    expect(html).toContain('href="/temas"');
    expect(html).toContain('<meta property="og:type" content="website" />');
  });

  it("hands a route it does not know to the SPA", async () => {
    expect((await app().request("/qualquer-coisa")).status).toBe(404);
    expect((await app().request("/datas/nao-e-um-ano")).status).toBe(404);
  });
});
