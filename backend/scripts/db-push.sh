#!/usr/bin/env bash
# Push the Prisma schema, then restore everything Prisma does not know about.
#
# `prisma db push` drops the generated `fts` column on every run: it is not in
# schema.prisma, so Prisma treats it as drift and removes it -- taking the GIN
# index and hybrid_search() with it. Always re-apply the raw SQL afterwards.
# Use this script rather than calling `prisma db push` directly.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DATABASE_URL:?DATABASE_URL must be set}"

prisma db push --skip-generate "$@"

echo "-> re-applying raw SQL (fts column, indexes, hybrid_search)"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f prisma/sql/001_hybrid_search.sql
echo "OK: schema + search objects in sync"
