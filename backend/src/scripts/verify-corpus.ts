import { access, readFile } from "node:fs/promises";
import { join } from "node:path";

import { loadSermons, MIN_SCORE, MIN_WORDS_MIN } from "../lib/corpus.ts";

/**
 * Checks that data/ is loadable before anything expensive touches it.
 *
 * The corpus is produced by `tools/corpus-update`, which is Python -- the
 * cleaning, scoring and CSV formatting all have to match the retired GPU
 * pipeline that wrote the original 455 rows. This is the seam where the
 * workspace's own data model gets the last word on that output: if a row does
 * not survive `loadSermons`, or promises a transcript that is not on disk, the
 * indexer would either skip it silently or crash a few hundred embeddings in.
 *
 *   pnpm verify:corpus
 */

type Problem = { name: string; reason: string };

async function main(): Promise<void> {
  const dataDir = process.env.CORPUS_DIR ?? "../data";
  const csvText = await readFile(join(dataDir, "metadata.csv"), "utf8");
  const { sermons, skipped } = loadSermons(csvText);

  const problems: Problem[] = [];
  for (const s of sermons) {
    try {
      await access(join(dataDir, "transcripts", s.transcriptFile));
    } catch {
      problems.push({ name: s.title, reason: `missing transcript ${s.transcriptFile}` });
    }
    if (!Number.isFinite(s.durationSec) || s.durationSec <= 0) {
      problems.push({ name: s.title, reason: `bad duration ${s.durationSec}` });
    }
    // A date that fell back to the epoch means neither the title nor the
    // publication timestamp was usable, which should now be impossible.
    if (s.date.getUTCFullYear() < 2015) {
      problems.push({ name: s.title, reason: `implausible date ${s.date.toISOString()}` });
    }
  }

  const newest = sermons.reduce((a, b) => (a.date > b.date ? a : b));
  console.log(
    `indexable sermons: ${sermons.length} (score > ${MIN_SCORE}, words/min > ${MIN_WORDS_MIN})`,
  );
  console.log(`newest: ${newest.date.toISOString().slice(0, 10)} — ${newest.title}`);

  // Named, not counted. A bare total is how nine truncated sermons went into
  // production unnoticed: the count moved and nobody could see what had left.
  console.log(`\nskipped by the loader: ${skipped.length}`);
  for (const s of skipped) console.log(`  ${s.name}: ${s.reason}`);

  if (problems.length > 0) {
    console.error(`\n${problems.length} problem(s):`);
    for (const p of problems) console.error(`  ${p.name}: ${p.reason}`);
    process.exit(1);
  }
  console.log("\ncorpus is loadable; every indexable sermon has its transcript");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
