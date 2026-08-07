import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { loadSermons, parseCsv } from "../lib/corpus.ts";
import { DATA_DIR } from "../lib/data-dir.ts";
import { writeCsv } from "../lib/facets/csv.ts";
import { fetchLiveness, type LivenessRow, parseLivenessCsv } from "../lib/podcast-feed.ts";

/**
 * Records which Spotify episodes still resolve.
 *
 *   pnpm --filter @ipp/backend check:spotify [--dry-run]
 *
 * Replaces a hard-coded date cutoff. The app used to hide every Spotify link
 * before 2022-01-01, on the theory that the old episodes had been retired by a
 * podcast-host migration. They were not: the church's feed is served by
 * SoundCloud, it is capped at 500 items, and every aggregator delists whatever
 * falls out of that window. See `../lib/podcast-feed.ts`.
 *
 * The date was only ever a proxy for feed membership, and a drifting one — it
 * hid 54 episodes that still worked, and would have gone on to *show* episodes
 * after they died. This writes the real answer instead, per episode.
 *
 * `data/facets/spotify_episodes.csv` is the ground truth and is committed. The
 * window rolls, so unlike the other derived files this one expires: it carries
 * `checked_at`, and the stage is re-runnable on its own without a corpus
 * update behind it.
 */
const COLUMNS = ["sermon_id", "spotify_id", "alive", "checked_at"];
const OUTPUT = join(DATA_DIR, "facets", "spotify_episodes.csv");
const DRY_RUN = process.argv.includes("--dry-run");

/** The previous run, or [] the first time — the dead-fraction guard needs a baseline. */
async function readPrevious(): Promise<LivenessRow[]> {
  try {
    return parseLivenessCsv(parseCsv(await readFile(OUTPUT, "utf8")));
  } catch {
    return [];
  }
}

async function main(): Promise<void> {
  const { sermons } = loadSermons(await readFile(join(DATA_DIR, "metadata.csv"), "utf8"));

  const { rows, feedItems } = await fetchLiveness({
    fetchFn: fetch,
    sermons,
    previous: await readPrevious(),
    checkedAt: new Date().toISOString().slice(0, 10),
  });

  const alive = rows.filter((r) => r.alive).length;
  console.log(
    `feed: ${feedItems} items · episodes: ${rows.length} (${alive} alive, ${rows.length - alive} aged out)`,
  );

  if (DRY_RUN) {
    console.log("--dry-run: not writing");
    return;
  }
  await writeFile(OUTPUT, writeCsv(COLUMNS, rows));
  console.log(`wrote ${OUTPUT}`);
}

await main();
