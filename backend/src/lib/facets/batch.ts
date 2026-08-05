/**
 * The shape both offline LLM passes share: work through a list of sermons a
 * few at a time, survive the ones that fail, and flush to disk as it goes.
 *
 * Extracted because `extract-scripture` and `label-topics` had grown the same
 * loop twice. The discipline it encodes is what makes a paid run safe to
 * interrupt: a failure never aborts the batch, and progress reaches disk long
 * before the end.
 */
const DEFAULT_CONCURRENCY = 4;

/** Reads `--limit N` off the command line; unbounded when absent. */
export function cliLimit(argv: string[] = process.argv): number {
  const at = argv.indexOf("--limit");
  const value = at !== -1 ? Number.parseInt(argv[at + 1] ?? "", 10) : Number.NaN;
  return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
}

type BatchOptions<T> = {
  /** How the item is named when it fails — a sermon title, not an id. */
  label: (item: T) => string;
  concurrency?: number;
  /** Called after roughly this many items, to persist what has been done. */
  flushEvery?: number;
  onFlush?: (done: number, total: number) => Promise<void>;
};

/**
 * Runs `worker` over every item, returning the failures rather than throwing.
 *
 * One unreachable sermon must not cost the whole run: it is reported and left
 * undone, and because both callers key their resume on what reached disk, the
 * next run picks it up with nothing else repeated.
 */
export async function runBatch<T>(
  items: T[],
  worker: (item: T) => Promise<void>,
  opts: BatchOptions<T>,
): Promise<string[]> {
  const concurrency = opts.concurrency ?? DEFAULT_CONCURRENCY;
  const flushEvery = opts.flushEvery ?? 40;
  const failures: string[] = [];
  let processed = 0;

  for (let i = 0; i < items.length; i += concurrency) {
    await Promise.all(
      items.slice(i, i + concurrency).map(async (item) => {
        try {
          await worker(item);
        } catch (err) {
          failures.push(`${opts.label(item)}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }),
    );

    processed += concurrency;
    if (processed % flushEvery < concurrency) {
      await opts.onFlush?.(Math.min(processed, items.length), items.length);
    }
  }

  return failures;
}

export function reportFailures(failures: string[]): void {
  if (failures.length === 0) return;

  console.log(`\n${failures.length} failed and will be retried on the next run:`);
  for (const failure of failures.slice(0, 10)) console.log(`  ${failure}`);
}
