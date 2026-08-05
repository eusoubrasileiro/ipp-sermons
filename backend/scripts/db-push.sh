#!/usr/bin/env bash
# Push the Prisma schema, then restore everything Prisma does not know about.
#
# `prisma db push` drops the generated `fts` column on every run: it is not in
# schema.prisma, so Prisma treats it as drift and removes it -- taking the GIN
# index and hybrid_search() with it. Always re-apply the raw SQL afterwards.
# Use this script rather than calling `prisma db push` directly.
#
# Every 0NN file after 000 is applied, in filename order, rather than a hard
# coded list: the list was already wrong once, and a migration that exists but
# is never applied fails silently -- the site just renders an empty facet.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DATABASE_URL:?DATABASE_URL must be set}"

prisma db push --skip-generate "$@"

echo "-> re-applying raw SQL (fts column, indexes, hybrid_search, facets)"
for sql in prisma/sql/[0-9][0-9][0-9]_*.sql; do
  case "$sql" in
    *000_schema.sql) continue ;;  # owned by `prisma db push` above
  esac
  echo "   $sql"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f "$sql"
done
echo "OK: schema + search objects in sync"
