-- Generated from schema.prisma via `prisma migrate diff`, then made
-- idempotent by hand: the migrate step re-runs on every deploy, and the
-- raw output has no IF NOT EXISTS, so a redeploy failed with exit 3.

-- CreateSchema
CREATE SCHEMA IF NOT EXISTS "public";

-- CreateExtension
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- CreateExtension
CREATE EXTENSION IF NOT EXISTS "vector";

-- CreateTable
CREATE TABLE IF NOT EXISTS "sermons" (
    "id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "artist" TEXT NOT NULL,
    "date" DATE NOT NULL,
    "duration_str" TEXT NOT NULL,
    "duration_sec" INTEGER NOT NULL,
    "sc_suffix_url" TEXT,
    "sp_suffix_url" TEXT,
    "score" DOUBLE PRECISION NOT NULL,
    "words" INTEGER NOT NULL,
    "sentences" INTEGER NOT NULL,
    "words_min" DOUBLE PRECISION NOT NULL,
    "sentences_min" DOUBLE PRECISION NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "sermons_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "sermon_chunks" (
    "id" TEXT NOT NULL,
    "sermon_id" TEXT NOT NULL,
    "chunk_index" INTEGER NOT NULL,
    "content" TEXT NOT NULL,
    "content_hash" TEXT NOT NULL,
    "embedding" halfvec(1536),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "sermon_chunks_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "suggestions" (
    "id" TEXT NOT NULL,
    "suggestion" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "suggestions_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "sermons_date_idx" ON "sermons"("date");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "sermons_artist_idx" ON "sermons"("artist");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "sermon_chunks_content_hash_idx" ON "sermon_chunks"("content_hash");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "sermon_chunks_sermon_id_chunk_index_key" ON "sermon_chunks"("sermon_id", "chunk_index");

-- AddForeignKey
DO $$ BEGIN
    ALTER TABLE "sermon_chunks" ADD CONSTRAINT "sermon_chunks_sermon_id_fkey" FOREIGN KEY ("sermon_id") REFERENCES "sermons"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

