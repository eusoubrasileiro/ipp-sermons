import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { PrismaClient } from "@prisma/client";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { readSermonTranscript } from "../src/lib/transcript.ts";

/**
 * Serving a whole sermon for reading.
 *
 * The text is not in the database — 20 MB of it already ships inside the image,
 * so the row carries only the file name and the reader goes to disk. That split
 * is what these tests pin: a row without a `transcriptFile`, or one naming a
 * file that is not there, must come back as "no transcript" rather than as a
 * 500, because both are states the corpus can legitimately be in between an
 * index run and a release.
 */

const dataDir = join(tmpdir(), `ipp-transcript-test-${process.pid}`);
const FILE = "01-01-2023 - Tito 2.txt";
const TEXT = "Como sempre, é um motivo de alegria estar aqui diante dessa querida igreja.";

function sermonRow(overrides: Record<string, unknown> = {}) {
  return {
    id: "123",
    title: "01-01-2023 - Tito 2",
    artist: "Reverendo Bruno Melo",
    date: new Date("2023-01-01T00:00:00Z"),
    durationStr: "1:02:39",
    scSuffixUrl: "tito-2",
    spSuffixUrl: "4rOoJ6Egrf8K2IrywzwOMk",
    spotifyAlive: true,
    words: 6424,
    transcriptFile: FILE,
    ...overrides,
  };
}

function prismaWith(row: ReturnType<typeof sermonRow> | null) {
  return { sermon: { findUnique: vi.fn().mockResolvedValue(row) } } as unknown as PrismaClient;
}

beforeAll(async () => {
  await mkdir(join(dataDir, "transcripts"), { recursive: true });
  await writeFile(join(dataDir, "transcripts", FILE), TEXT, "utf8");
});

describe("readSermonTranscript", () => {
  it("returns the whole text with the sermon's metadata", async () => {
    const got = await readSermonTranscript(prismaWith(sermonRow()), dataDir, "123");

    expect(got).toEqual({
      id: "123",
      title: "01-01-2023 - Tito 2",
      artist: "Reverendo Bruno Melo",
      date: "2023-01-01",
      durationStr: "1:02:39",
      soundcloudUrl: "https://soundcloud.com/ipperegrinos/tito-2",
      spotifyUrl: "https://open.spotify.com/episode/4rOoJ6Egrf8K2IrywzwOMk",
      words: 6424,
      text: TEXT,
    });
  });

  it("builds the play links the same way search does", async () => {
    // `sc_suffix_url` is a track slug, not a URL, and a dead Spotify episode is
    // suppressed rather than linked. Both rules live in shared/audio-urls; a
    // second copy here is how the play links 404'd once already.
    const row = sermonRow({ spotifyAlive: false, scSuffixUrl: null });
    const got = await readSermonTranscript(prismaWith(row), dataDir, "123");

    expect(got?.soundcloudUrl).toBeNull();
    expect(got?.spotifyUrl).toBeNull();
  });

  it("returns null for a sermon that is not in the database", async () => {
    expect(await readSermonTranscript(prismaWith(null), dataDir, "nope")).toBeNull();
  });

  it("returns null when the row carries no transcript file", async () => {
    // Every row indexed before the column existed looks like this until the
    // next index run fills it in. A reading page that 404s is correct; a 500 is
    // not.
    const row = sermonRow({ transcriptFile: null });
    expect(await readSermonTranscript(prismaWith(row), dataDir, "123")).toBeNull();
  });

  it("returns null when the named file is not on disk", async () => {
    // The database and data/ are shipped together but pruned separately, so a
    // row can name a transcript a later corpus update removed.
    const row = sermonRow({ transcriptFile: "nao-existe.txt" });
    expect(await readSermonTranscript(prismaWith(row), dataDir, "123")).toBeNull();
  });

  it("refuses a transcript file that climbs out of data/transcripts", async () => {
    // The name comes from the database rather than the URL, so this is defence
    // in depth -- but the value ends up in a path join, and the corpus is
    // rebuilt from a CSV that no schema validates.
    const row = sermonRow({ transcriptFile: "../../../etc/passwd" });
    expect(await readSermonTranscript(prismaWith(row), dataDir, "123")).toBeNull();
  });
});
