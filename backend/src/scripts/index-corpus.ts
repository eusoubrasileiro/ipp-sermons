import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import pkg, { type PrismaClient as PrismaClientType } from "@prisma/client";

// @prisma/client is CommonJS: `import { Prisma }` resolves under the dev
// loader but not in the compiled ESM build, where named-export detection
// misses it. Destructure from the default export instead.
const { Prisma, PrismaClient } = pkg;

import { chunkHash, chunkText } from "../lib/chunking.ts";
import { loadSermons, readTranscript, type SermonRecord } from "../lib/corpus.ts";
import { createOpenRouterEmbeddings, EMBEDDING_DIMS, toVectorLiteral } from "../lib/embeddings.ts";
import { createUsageMeter } from "../lib/usage.ts";

/**
 * One-shot corpus indexer: transcripts on disk -> chunks + embeddings in Postgres.
 *
 * Idempotent. Each chunk is keyed by a content hash, so re-running skips work
 * that is already done and a run interrupted halfway resumes cleanly. That
 * matters: this makes thousands of paid API calls, and crashing at 90% should
 * not mean paying for the first 90% again.
 *
 *   pnpm index                # whole corpus
 *   pnpm index --limit 5      # smoke test
 *   pnpm index --force        # re-embed even unchanged chunks
 */

const BATCH_SIZE = 64;

type Args = { limit?: number; force: boolean; dataDir: string };

function parseCliArgs(): Args {
  const { values } = parseArgs({
    options: {
      limit: { type: "string" },
      force: { type: "boolean", default: false },
      "data-dir": { type: "string" },
    },
  });

  return {
    limit: values.limit ? Number.parseInt(values.limit, 10) : undefined,
    force: values.force ?? false,
    dataDir: values["data-dir"] ?? process.env.CORPUS_DIR ?? "../data",
  };
}

async function upsertSermon(prisma: PrismaClientType, s: SermonRecord): Promise<void> {
  const row = {
    title: s.title,
    artist: s.artist,
    date: s.date,
    durationStr: s.durationStr,
    durationSec: s.durationSec,
    scSuffixUrl: s.scSuffixUrl,
    spSuffixUrl: s.spSuffixUrl,
    score: s.score,
    words: s.words,
    sentences: s.sentences,
    wordsMin: s.wordsMin,
    sentencesMin: s.sentencesMin,
  };
  await prisma.sermon.upsert({ where: { id: s.id }, create: { id: s.id, ...row }, update: row });
}

async function main(): Promise<void> {
  const args = parseCliArgs();
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is not set");

  const prisma = new PrismaClient();
  const meter = createUsageMeter();
  const embeddings = createOpenRouterEmbeddings({ apiKey, onUsage: meter.record });

  const csvText = await readFile(join(args.dataDir, "metadata.csv"), "utf8");
  const { sermons, skipped } = loadSermons(csvText);

  const selected = args.limit ? sermons.slice(0, args.limit) : sermons;

  console.log(`corpus: ${sermons.length} sermons eligible, ${skipped.length} skipped`);
  if (args.limit) console.log(`--limit ${args.limit}: indexing ${selected.length}`);

  let chunksSeen = 0;
  let chunksEmbedded = 0;
  let chunksSkipped = 0;

  for (const [i, sermon] of selected.entries()) {
    let transcript: string;
    try {
      transcript = await readTranscript(args.dataDir, sermon.transcriptFile);
    } catch {
      console.warn(`  ! missing transcript, skipping: ${sermon.transcriptFile}`);
      continue;
    }

    await upsertSermon(prisma, sermon);

    const chunks = chunkText(transcript);
    chunksSeen += chunks.length;

    // Work out which chunks actually need embedding before spending anything.
    const pending: { index: number; content: string; hash: string }[] = [];
    for (const [index, content] of chunks.entries()) {
      const hash = chunkHash(sermon.id, index, content);
      if (!args.force) {
        // `embedding` is an Unsupported column, so it cannot appear in a
        // select at all -- not even as `false`. Ask only for the hash.
        const existing = await prisma.sermonChunk.findUnique({
          where: { sermonId_chunkIndex: { sermonId: sermon.id, chunkIndex: index } },
          select: { contentHash: true },
        });
        if (existing?.contentHash === hash) {
          chunksSkipped++;
          continue;
        }
      }
      pending.push({ index, content, hash });
    }

    for (let b = 0; b < pending.length; b += BATCH_SIZE) {
      const batch = pending.slice(b, b + BATCH_SIZE);
      const vectors = await embeddings.embed(batch.map((c) => c.content));

      for (const [j, chunk] of batch.entries()) {
        const vec = vectors[j];
        if (!vec) throw new Error(`missing embedding for chunk ${chunk.index}`);

        // Prisma has no type for halfvec, so the vector goes in via raw SQL.
        // Everything else stays parameterised.
        //
        // The dimension in the cast must be interpolated, not bound: Postgres
        // requires type modifiers to be literal constants, so `halfvec($1)`
        // fails with "type modifiers must be simple constants or identifiers".
        // EMBEDDING_DIMS is our own integer constant, not user input.
        await prisma.$executeRaw`
          INSERT INTO sermon_chunks (id, sermon_id, chunk_index, content, content_hash, embedding)
          VALUES (gen_random_uuid()::text, ${sermon.id}, ${chunk.index}, ${chunk.content},
                  ${chunk.hash}, ${toVectorLiteral(vec)}::halfvec(${Prisma.raw(String(EMBEDDING_DIMS))}))
          ON CONFLICT (sermon_id, chunk_index) DO UPDATE
            SET content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                embedding = EXCLUDED.embedding`;

        chunksEmbedded++;
      }
    }

    const n = String(i + 1).padStart(3);
    console.log(
      `  [${n}/${selected.length}] ${sermon.title.slice(0, 58)} ` +
        `(${chunks.length} chunks, ${pending.length} new)`,
    );
  }

  console.log(
    `\ndone: ${chunksEmbedded} chunks embedded, ${chunksSkipped} unchanged, ${chunksSeen} total`,
  );
  // Reported by OpenRouter, not multiplied out of a price list here: the list
  // goes stale without anyone noticing, and the token count was itself a guess
  // from word counts.
  const spend = meter.summary();
  if (spend) console.log(spend);

  await prisma.$disconnect();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
