# ipp-sermons

Busca em português sobre os sermões transcritos da Igreja Presbiteriana
Peregrinos — <https://ipp-sermons.amiticia.cc>

Pergunte em linguagem natural ("briga na igreja", "Eclesiastes tempo de
plantar") e receba os sermões que respondem, com o trecho que casou e o link
para ouvir no SoundCloud ou Spotify.

## Stack

pnpm workspace, Node 24. `shared/` (Zod) · `backend/` (Hono + Prisma +
Postgres/pgvector) · `frontend/` (Vite + React 19 + Tailwind).

Retrieval is hybrid: lexical (`tsvector`, Portuguese with accent folding) and
semantic (`halfvec(1536)`, HNSW) fused by Reciprocal Rank Fusion inside a
Postgres function, then reranked by a cross-encoder.

## Running it

```bash
# install and build the shared package everything else depends on
pnpm install
pnpm --filter @ipp/shared build
pnpm --filter @ipp/backend exec prisma generate

# Postgres + pgvector on :5439, then schema + the raw search SQL
pnpm db:up
pnpm db:push

# index the corpus (needs OPENROUTER_API_KEY; --limit 5 for a smoke run)
pnpm index

pnpm dev
```

Copy `.env.example` to `.env` first. Architecture, operational traps and the
contribution gates are documented in [`CLAUDE.md`](./CLAUDE.md).

## Corpus

456 sermons, ~20 MB of transcripts under `data/`. Audio is never hosted here.
Transcription runs offline in `tools/corpus-update/` (Python + WhisperX).
