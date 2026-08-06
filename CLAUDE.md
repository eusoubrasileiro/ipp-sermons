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
        │
        ▼
frontend/ Vite + React 19, built into backend/public/ and served same-origin
```

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
| `shared/src/` | Zod schemas + types shared by API and UI. Built to `dist/` first; everything else depends on it. |
| `backend/src/lib/` | `search` (retrieval), `embeddings`, `rerank`, `llm` + `openrouter` (offline passes), `corpus` (CSV + transcript reader), `audio-urls` |
| `backend/src/lib/facets/` | Facet derivation: `bible`, `parse-title`, `parse-scripture`, `cluster`, `series-taxonomy`, `extract-prompt`, `topics`, `batch`, `csv`, `slugify` |
| `backend/src/lib/browse/` | `facets` (index tree), `counts` (adjusted for active filters), `list` (filtered listing), `query` (shared param parsing) |
| `backend/src/scripts/` | `index-corpus`, `eval-golden`, and the facet pipeline below |
| `backend/prisma/` | `schema.prisma` + `sql/` — **the sql files ARE the migrations** |
| `frontend/src/` | Vite + React 19 + Tailwind SPA, Portuguese UI. Filter state lives in the URL (`lib/facet-params.ts`). |
| `data/` | Corpus: `metadata.csv`, `transcripts/`, `preacher_names.txt` |
| `data/facets/` | Derived ground truth, committed and reviewed: `bible_books`, `series`, `taxonomy`, `sermon_facets`, `sermon_scriptures`, `scripture_llm`, `sermon_topics` |
| `tools/corpus-update/` | Python: discover → fetch → transcribe → clean. Runs offline, feeds `data/`. |
| `deploy/` | Production compose for the Hostinger VPS |
| `scripts/` | Dev harness (quality-gate, security-review, dispatch-worktree) |
| `archive/` | Retired GPU-era Python. Reference only — never revive. |

## Development

```bash
pnpm install
pnpm --filter @ipp/shared build              # everything else needs its dist/
pnpm --filter @ipp/backend exec prisma generate   # client is generated, not committed

pnpm db:up                                   # Postgres+pgvector on :5439
pnpm db:push                                 # schema + raw SQL — NOT `prisma db push`
pnpm index                                   # index the corpus (--limit 5 to smoke it)
pnpm index:facets                            # load data/facets/ — free, seconds, no API

pnpm dev                                     # backend + frontend
pnpm test                                    # unit tests, no DB, no network
pnpm test:coverage                           # writes coverage-summary.json for the gate
pnpm typecheck
pnpm lint
pnpm eval                                    # golden query set vs the REAL db + API
pnpm quality-gate                            # metric ratchet vs quality-baseline.json
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

9. **Spotify links are suppressed for pre-2022 sermons** (`SPOTIFY_LINKS_ALIVE_FROM`
   in `audio-urls.ts`). Roughly a third of the episode ids no longer resolve and
   every dead one is from 2019–2021 — the episodes were retired upstream, most
   likely a podcast-host migration. The ids match both Spotify's API at scrape
   time and an independent 2025 scrape, so this is a workaround for dead
   upstream data, not an app bug or a corrupt column. SoundCloud covers 100% of
   the corpus. Remove the constant and its guard if the old ids are re-scraped.

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

Three services: `db` (pgvector/pg16), `migrate` (one-shot `postgres:16-alpine`
applying the committed SQL, idempotent), `app`. Memory limits are not decorative
— the box has ~2 GB free of 7.8 GB, plus 4 GB of swap added for this deployment.

Rollback: pin `IMAGE_TAG` in `.env` to the previous digest and
`docker compose up -d`.

## Constraints

- **UI text is Portuguese.** Code, comments and commit messages are English.
- **OpenRouter is the single LLM credential** — embeddings and reranking both.
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
