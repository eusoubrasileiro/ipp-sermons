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

Copy `.env.example` to `.env`, then follow the Development section of
[`CLAUDE.md`](./CLAUDE.md) — it also carries the architecture, the operational
traps and the contribution gates.

## Corpus

~610 sermons, ~27 MB of transcripts under `data/`. Audio is never hosted here.
Transcription runs offline in `tools/corpus-update/` (Python + WhisperX).
