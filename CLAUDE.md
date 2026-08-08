# CLAUDE.md

This project runs under the [Anti-Vibe Harness](../../amiticia/repositories/standards/standards.md) —
AmiticIA's AI-agent discipline stack (dual git gate, ratcheted quality-gate, LLM
reviewer, worktree dispatch). TypeScript specifics:
`standards/typescript.md`.

## Project Overview

Portuguese-language search over the transcribed sermon archive of the Igreja
Presbiteriana Peregrinos. A visitor types a question in natural Portuguese
("briga na igreja", "Eclesiastes tempo de plantar") and gets the sermons that
answer it, with a playable SoundCloud/Spotify link and the passage that matched.

**Live** at https://ipp-sermons.amiticia.cc. It replaces a retired GPU pipeline:
transcription still happens offline (`tools/corpus-update`, Python + WhisperX),
but retrieval is now API-only — no local models, no GPU, one VPS container.

Corpus: 456 sermons, ~20,000 chunks, ~20 MB of text committed under `data/`.
The 77 GB of audio lives on SoundCloud and is never hosted here.

## Architecture

```
data/metadata.csv + data/transcripts/*.txt
        │  pnpm index         (chunk → embed via OpenRouter → upsert)
        │  pnpm index:facets  (load the committed CSVs in data/facets/)
        ▼
Postgres 16 + pgvector          ← the whole retrieval engine lives here
  sermons, sermon_chunks(embedding halfvec(1536), fts tsvector)
  bible_books, series, topics, sermon_scriptures, sermon_topics
  hybrid_search()  = BM25-ish lexical ⊕ vector, fused by RRF, facet-filtered
        │
        ▼
backend/  Hono   POST /api/search  → hybrid_search() → cross-encoder rerank
                 GET  /api/facets  → the browse index tree
                 GET  /api/facets/counts → the same, narrowed to active filters
                 GET  /api/sermons → filtered listing, no query, no model
                 GET  /sermao/:id, /biblia/…, /sitemap.xml → prerendered HTML
        │
        ▼
frontend/ Vite + React 19, built into backend/public/ and served same-origin
```

### Why the server renders HTML at all

The SPA fallback answered **every** unmatched GET with the same empty
`index.html`, so 560 sermons — 3.6 million words of exactly the long-tail
Portuguese prose a search engine rewards — were one indexable URL. `lib/seo/`
rewrites the built shell per request: the page's own `<title>`, a description
cut from the sermon's opening words, canonical + Open Graph tags, and the text
itself inside `<div id="root">`.

It is **string injection, not `renderToString`**. The runtime image ships
`backend/dist` and `backend/public` only; React and the JSX pages are not in it,
and `main.tsx` calls `createRoot` (not `hydrateRoot`), which clears its container
before the first render — so the injected body never has to match React's output
byte for byte, only carry the same words. Anything it cannot build (missing
sermon, unknown facet slug, database down, frontend never built) calls `next()`
and lets the static middleware answer exactly as before.

One container serves API and SPA, so there is no CORS and no second origin in
production. `shared/` holds the Zod schemas both sides validate against.

### Retrieval design

Two arms, fused by **Reciprocal Rank Fusion** (k=60), plus a third arm on sermon
titles, then a cross-encoder rerank:

- **Lexical** (`ts_rank_cd` over a generated `fts` column, `pt_unaccent` config).
  Earns its place because the corpus is full of proper nouns and scripture
  references that embeddings blur — "Eclesiastes 3.2b-3a", "Jonathan Edwards".
- **Semantic** (`halfvec(1536)` HNSW, cosine). `gemini-embedding-001` truncated
  3072→1536 via Matryoshka and re-normalised.
- **Title** arm, weighted 1.5×. A preacher rarely says the title out loud, so
  "O sétimo mandamento" ranked 52nd for its own title until titles became their
  own arm.
- **Rerank** (`cohere/rerank-4-pro`) over 40 candidates. Every failure mode
  degrades to the RRF order — a slow or unreachable reranker must never turn a
  working search into an error page.

**The fusion is deliberately SQL, not LangChain.** `PGVectorStore` cannot express
a BM25 + RRF fusion; it exposes vector similarity and nothing to fuse it with.
Doing it in application code would mean shipping both candidate lists over the
wire to rank them. `hybrid_search()` in `backend/prisma/sql/001_hybrid_search.sql`
is the honest place for it. Do not "modernise" this into a framework call.

## Code Organization

| Path | Purpose |
|------|---------|
| `shared/src/` | Zod schemas + types shared by API and UI, plus `audio-urls` (both sides build play links), `format` (title/date/`<title>`) and `paragraphs` (transcript splitting) — the server prerenders these and the SPA renders them again, so one copy. Built to `dist/` first; everything else depends on it. |
| `backend/src/lib/` | `search` (retrieval), `embeddings`, `rerank`, `llm` + `openrouter` (offline passes), `corpus` (CSV + transcript reader), `podcast-feed` (Spotify liveness) |
| `backend/src/lib/facets/` | Facet derivation: `bible`, `parse-title`, `parse-scripture`, `cluster`, `series-taxonomy`, `extract-prompt`, `topics`, `batch`, `csv`, `slugify` |
| `backend/src/lib/browse/` | `facets` (index tree), `counts` (adjusted for active filters), `list` (filtered listing), `query` (shared param parsing) |
| `backend/src/lib/seo/` | Server-rendered HTML for crawlers: `html` (escaping + shell injection), `shell` (the built `index.html`), `sermon-page`, `listing-page`, `browse-pages`, `sitemap`, `routes`, `layout`, `site` |
| `backend/src/scripts/` | `index-corpus`, `eval-golden`, and the facet pipeline below |
| `backend/prisma/` | `schema.prisma` + `sql/` — **the sql files ARE the migrations** |
| `frontend/src/` | Vite + React 19 + Tailwind SPA, Portuguese UI. Filter state lives in the URL (`lib/facet-params.ts`). |
| `data/` | Corpus: `metadata.csv`, `transcripts/`, `preacher_names.txt` |
| `data/facets/` | Derived ground truth, committed and reviewed: `bible_books`, `series`, `taxonomy`, `sermon_facets`, `sermon_scriptures`, `scripture_llm`, `sermon_topics`, `spotify_episodes` |
| `tools/corpus-update/` | Python: discover → fetch → transcribe → clean. Runs offline, feeds `data/`. |
| `deploy/` | Production compose for the Hostinger VPS |
| `scripts/` | Dev harness (quality-gate, security-review, dispatch-worktree) + `corpus-update.sh`, which owns the order of a corpus update |
| `archive/` | Retired GPU-era Python. Reference only — never revive. |
| `docs/` | Work that is designed but deliberately not done yet, and why |

## Development

```bash
pnpm install
pnpm --filter @ipp/shared build              # everything else needs its dist/
pnpm --filter @ipp/backend exec prisma generate   # client is generated, not committed

pnpm db:up                                   # Postgres+pgvector on :5439
pnpm db:push                                 # schema + raw SQL — NOT `prisma db push`
pnpm index                                   # index the corpus (--limit 5 to smoke it)
pnpm index:facets                            # load data/facets/ — free, seconds, no API
pnpm check:spotify                           # refresh which Spotify episodes still resolve

pnpm dev                                     # backend + frontend
pnpm test                                    # unit tests, no DB, no network
pnpm test:coverage                           # writes coverage-summary.json for the gate
pnpm typecheck
pnpm lint
pnpm eval                                    # golden query set vs the REAL db + API
pnpm quality-gate                            # metric ratchet vs quality-baseline.json

pnpm corpus:update                           # SoundCloud → data/ → facets → prod
pnpm corpus:update --review                  # same, stopping at each checkpoint
```

`pnpm eval` is the only thing that proves the search is any good. Unit tests
prove the plumbing. Run it after any change to `hybrid_search()`, the embedding
model, chunking, or rerank — an 8/8 recall@10 that silently drops to 5/8 is
invisible to every other gate in this repo.

## Traps worth knowing

Each of these cost real debugging time. They are not obvious from the code.

1. **`prisma db push` drops the `fts` column every run.** The generated tsvector
   is not in `schema.prisma`, so Prisma reads it as drift and removes it —
   taking the GIN index and `hybrid_search()` with it. Always
   `pnpm db:push` (`backend/scripts/db-push.sh`), which re-applies the raw SQL.

2. **`websearch_to_tsquery` ANDs bare terms.** "briga na igreja" became
   `'brig' & 'igrej'` and matched only chunks containing both stems — silently
   zeroing the lexical arm for most multi-word queries and leaving "hybrid"
   search running on the vector arm alone. `ipp_to_tsquery()` rewrites the
   top-level ANDs to ORs; quoted phrases keep their `<->` operators.

3. **`@prisma/client` is CommonJS.** `import { PrismaClient } from "@prisma/client"`
   resolves under the dev loader and throws in the compiled ESM build, where
   named-export detection misses it. Use a default import and destructure:
   `import pkg from "@prisma/client"; const { PrismaClient } = pkg;`

4. **Prisma sends JS numbers as bigint.** SQL function arguments need an explicit
   cast — `${candidateCount}::int` — or Postgres cannot resolve the overload.

5. **`halfvec($1)` is illegal.** Postgres requires type modifiers to be literal,
   so the vector cast is string-interpolated via `Prisma.raw` rather than bound.
   The interpolated values are ours, never user input; `query` stays bound.

6. **Migrations are committed SQL applied by a `postgres:16-alpine` sidecar,
   deliberately not the Prisma CLI.** The CLI is a devDependency that `--prod`
   strips; pulling it back drags `@prisma/engines` into the runtime image, and
   `npx prisma` ignores the pinned version — it fetched Prisma 7 mid-deploy, a
   major that had dropped `--skip-generate`. `000_schema.sql` was hand-edited to
   be idempotent because `migrate diff` emits unguarded `CREATE`/`ADD CONSTRAINT`.

7. **Truncated Matryoshka vectors are not unit length** (‖v‖ ≈ 0.697). pgvector's
   cosine operator assumes they are, so `normalize()` in `embeddings.ts` is
   load-bearing: without it nothing errors and every similarity is quietly wrong.

8. **`sc_suffix_url` is a track slug, not a URL.** It is yt-dlp's
   `webpage_url_basename`, so the SoundCloud channel has to be prepended —
   `https://soundcloud.com/ipperegrinos/<slug>`. Omitting it 404s every play
   link, which shipped once. `sp_suffix_url` is a bare 22-char Spotify episode
   id and needs no show context. Both are rebuilt at read time in
   `backend/src/lib/audio-urls.ts`; nothing stores a full URL.

9. **Spotify episodes die when they age out of the podcast feed, not when they
   get old.** Every platform the church publishes to reads one URL — the
   SoundCloud-generated RSS feed — and it is **capped at 500 items**. Whatever
   falls out of that window gets delisted, so the episode 404s while its id
   stays perfectly valid. The window *rolls*: ~50–100 new sermons a year push
   the oldest off, so the answer expires on its own.

   This replaced a `SPOTIFY_LINKS_ALIVE_FROM = "2022-01-01"` date cutoff that
   was a proxy for feed membership, and a drifting one — it hid 52 episodes of
   2021 that still worked, and would eventually have shown episodes already
   dead. `pnpm check:spotify` records the truth per episode in
   `data/facets/spotify_episodes.csv`; `spotify_alive` on `sermons` is what the
   app reads. SoundCloud covers 100% of the corpus and is never suppressed.
   Background for the church's IT contact: `docs/spotify-feed-window.md`.

10. **The prerendered HTML must never carry a `max-age`.** It names the bundle
    in `<script src="/assets/index-<hash>.js">`, and every release changes that
    hash. A browser holding the document asks the new container for the old
    file, the SPA catch-all answers with `index.html`, and the browser refuses
    it as a module on MIME grounds — a blank site for the length of the
    max-age. `HTML_CACHE_CONTROL` in `lib/seo/routes.ts` is
    `max-age=0, must-revalidate` for exactly this; `sitemap.xml` and
    `robots.txt` name no hashed asset and keep their hour.

11. **`String.replace` with a replacement *string* expands `$&`, `` $` `` and
    `$'`.** The shell injection interpolates sermon titles and 40 KB of
    transcribed speech, so every replace in `lib/seo/html.ts` passes a
    *function* instead. With a string, a sermon titled with a `$&` would splice
    the surrounding `index.html` into its own page.

12. **The SPA now has to maintain `document.title`.** It never did — the title
    was one constant for the whole site. Since the server writes the page's real
    title, a client-side navigation away would leave the tab advertising a page
    the visitor left. `lib/useDocumentTitle.ts` keeps what the server wrote for
    the page actually landed on and resets it on any navigation away.

## Critical files

Protected by the `ask` tier in `.claude/settings.json` and by the critical-paths
list in `scripts/security-review.mjs`. **The three must stay in sync** — if they
drift, the reviewer rejects what the settings allow, or worse, the reverse.

| Path | Why |
|------|-----|
| `**/*.md` | Every markdown file (company rule, `standards.md` §8). A stray spec reads as authoritative to the next agent. |
| `.gitignore` | The only thing keeping rendered artifacts and secrets out of git. |
| `.husky/**` | Bypassing the hooks defeats the whole stack. |
| `.claude/settings.json` | Loosening it lets agents edit critical files unattended. |
| `commitlint.config.cjs` | Commit-message contract the reviewer keys off. |
| `quality-baseline.json` | The metric floor. Loosening it without a source-level improvement hides a regression. |
| `scripts/quality-gate.mjs`, `scripts/security-review.mjs`, `scripts/lib/**` | The gate itself. |
| `scripts/dispatch-worktree.sh`, `scripts/cleanup-worktrees.sh` | Worktree provisioning and teardown. |
| `backend/prisma/**` | The SQL files ARE the migrations — a deploy applies them verbatim to live data, with no Prisma CLI to catch drift. |
| `backend/scripts/db-push.sh` | The only safe local schema sync (see trap 1). |
| `backend/test/golden/queries.json` | The retrieval eval contract. Editing it to make a ranking regression pass is precisely the failure it exists to catch. |
| `deploy/**`, `Dockerfile`, `docker-compose.yml` | Production topology: Traefik labels, TLS resolver, and the memory limits that keep a 7.8 GB VPS alive. |
| `data/**` | The corpus. Ground truth, not code. |

## Testing discipline (TDD — mandatory)

Red-Green-Refactor. Failing test first, then implement, then refactor.

| Change | Test written FIRST |
|---|---|
| New pure helper | Unit test |
| New/changed API route | Route test driving the real Hono app with stubbed deps |
| Bug fix | Regression test reproducing the bug |
| Retrieval change (SQL, chunking, model, rerank) | A golden query in `backend/test/golden/queries.json`, then `pnpm eval` |
| Refactor | Existing tests green first |

Unit tests touch neither DB nor network — `createApp()` takes its dependencies
so the real routes run against stubs. Coverage is ratcheted through
`quality-baseline.json`, not hard-coded thresholds.

## Multi-agent dispatch

Sub-agents never run in the parent worktree. `pnpm dispatch <slug>` creates
`.claude/worktrees/<slug>/` on branch `agent/<slug>` with hash-derived ports, its
own Postgres database (`ipp-agent-<slug>`), a symlinked `.env`, a built
`@ipp/shared`, a generated Prisma client, and a stamped `.claude/AGENT.md`.

The leader writes `.claude/plans/<slug>.md` **before** dispatching — the script
hard-fails without it, and the plan becomes the sub-agent's contract.

```bash
pnpm dispatch my-task
pnpm dispatch:cleanup --slug my-task           # branch deleted only if merged
pnpm dispatch:cleanup --slug my-task --force
```

Running several agents in the *same* working tree is the anti-pattern this
exists to prevent: one agent's half-finished change turns everyone else's
pre-commit red, and nobody can commit.

## Gates

`.husky/pre-commit` — lint-staged (Biome) → `pnpm typecheck` → `pnpm test:coverage`
→ `pnpm quality-gate`.

`.husky/pre-push` — typecheck → lint → coverage → `pnpm quality-gate` →
`node scripts/security-review.mjs` (Sonnet reviewer, fail-closed) →
`show-review-log.mjs` so the verdicts are visible before the push completes.

`.husky/commit-msg` — commitlint, conventional commits. The reviewer keys off
`fix:`/`bug:`/`hotfix:` to require a regression test in the same commit.

Never `--no-verify`. A failing hook means fixing the root cause, never editing
the gate to agree with the code.

## Deployment

Image `ghcr.io/eusoubrasileiro/ipp-sermons` → Hostinger VPS at
`/opt/amiticia/ipp-sermons/`, which holds only `.env`, `docker-compose.yml` and
`sql/`. There is no checkout on the VPS. Traefik v3 on the external
`network_public` terminates TLS and routes `ipp-sermons.amiticia.cc`.

Four services, in dependency order: `db` (pgvector/pg16), `migrate` (one-shot
`postgres:16-alpine` applying the committed SQL, idempotent), `index` and
`facets` (one-shot, in that order — `index-facets` filters its rows to the
sermon ids already in the database), `app`. Memory limits are not decorative —
the box has ~2 GB free of 7.8 GB, plus 4 GB of swap added for this deployment.

Rollback: pin `IMAGE_TAG` in `.env` to the previous digest and
`docker compose up -d`.

## Constraints

- **UI text is Portuguese.** Code, comments and commit messages are English.
- **OpenRouter is the single LLM credential** — embeddings and reranking both.
- **`PUBLIC_BASE_URL`** sets the origin in canonical and Open Graph URLs. It
  defaults to the production origin, so nothing has to be set to deploy; set it
  when serving the site under any other name.
- Embeddings are **1536-d, not 3072**: pgvector caps HNSW on `vector` at 2000
  dimensions. Changing dimension means re-indexing the whole corpus (paid).
- The corpus is filtered to alignment `score > 50`, inherited from the GPU
  pipeline. Transcripts below that are noise.
- `data/metadata.csv` column names are inherited from `sermons_ai/doc_preproc.py`.
  Quirks there are historical, not chosen — don't "clean them up".

## What we won't build

- **No LangChain / LlamaIndex for retrieval.** The reason hybrid search is in SQL
  is that the framework abstraction cannot express it (see Architecture).
- **No self-hosted models.** The whole point of the rewrite was to retire the GPU
  box. Embeddings and reranking go over an API or not at all.
- **No user accounts, no auth.** Public read-only search over a public archive.
- **No audio hosting.** SoundCloud and Spotify own the 77 GB; we link to it.
- **No admin UI for the corpus.** Corpus updates are a batch pipeline
  (`tools/corpus-update`), reviewed by a human, committed to git.
