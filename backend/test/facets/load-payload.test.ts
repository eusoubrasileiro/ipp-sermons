import { describe, expect, it } from "vitest";
import {
  scripturePayload,
  spotifyPartition,
  topicPayload,
} from "../../src/lib/facets/load-payload.ts";

/**
 * The rows a facet load is about to write, computed before anything is deleted.
 *
 * `index-facets.ts` replaces sermon_scriptures and sermon_topics wholesale --
 * correctly, because a chapter that stopped being derived must stop being
 * listed. The bug was the order: it deleted, then built the payload, then
 * checked whether the payload was empty. A column rename or a slug drift
 * therefore emptied the table and only then threw, inside the one-shot `facets`
 * service, on every production deploy.
 *
 * Pulling the payload out is what lets the guard run first.
 */
const SERMONS = new Set(["876988777", "732119686"]);
const BOOKS = new Set(["mateus", "genesis"]);
const TOPICS = new Set(["perdao-e-reconciliacao"]);

const scriptureRow = (over: Record<string, string> = {}) => ({
  sermon_id: "876988777",
  book_slug: "mateus",
  chapter: "6",
  verse_start: "12",
  verse_end: "15",
  source: "titulo",
  is_primary: "true",
  ...over,
});

describe("scripturePayload", () => {
  it("maps a well-formed row onto the database shape", () => {
    expect(scripturePayload([scriptureRow()], SERMONS, BOOKS)).toEqual([
      {
        sermonId: "876988777",
        bookSlug: "mateus",
        chapter: 6,
        verseStart: 12,
        verseEnd: 15,
        source: "titulo",
        isPrimary: true,
      },
    ]);
  });

  it("drops a row whose sermon is not indexed yet", () => {
    // index-corpus has to run first; a sermon absent from the database means
    // this row would violate the foreign key.
    const rows = [scriptureRow(), scriptureRow({ sermon_id: "999" })];
    expect(scripturePayload(rows, SERMONS, BOOKS)).toHaveLength(1);
  });

  it("drops a row naming a book that does not exist", () => {
    const rows = [scriptureRow(), scriptureRow({ book_slug: "hezekiah" })];
    expect(scripturePayload(rows, SERMONS, BOOKS)).toHaveLength(1);
  });

  it("reads a whole-chapter reference as null verses", () => {
    const [row] = scripturePayload(
      [scriptureRow({ verse_start: "", verse_end: "" })],
      SERMONS,
      BOOKS,
    );
    expect(row).toMatchObject({ verseStart: null, verseEnd: null });
  });

  it("throws rather than return nothing when every row filters out", () => {
    // The failure this exists to catch: a rename makes every row unmatched, the
    // load reports zero, the deploy succeeds, and every scripture page renders
    // empty. Throwing here happens before the delete, so the table survives.
    expect(() => scripturePayload([scriptureRow({ sermon_id: "999" })], SERMONS, BOOKS)).toThrow(
      /none matched/,
    );
  });

  it("accepts a genuinely empty file", () => {
    expect(scripturePayload([], SERMONS, BOOKS)).toEqual([]);
  });
});

describe("topicPayload", () => {
  const topicRow = (over: Record<string, string> = {}) => ({
    sermon_id: "876988777",
    topico_slug: "perdao-e-reconciliacao",
    confianca: "1",
    ...over,
  });

  it("maps a well-formed row onto the database shape", () => {
    expect(topicPayload([topicRow()], SERMONS, TOPICS)).toEqual([
      { sermonId: "876988777", topicSlug: "perdao-e-reconciliacao", confidence: 1 },
    ]);
  });

  it("drops a row naming a topic outside the taxonomy", () => {
    const rows = [topicRow(), topicRow({ topico_slug: "inventado" })];
    expect(topicPayload(rows, SERMONS, TOPICS)).toHaveLength(1);
  });

  it("falls back to full confidence when the column is unreadable", () => {
    expect(topicPayload([topicRow({ confianca: "" })], SERMONS, TOPICS)[0]).toMatchObject({
      confidence: 1,
    });
  });

  it("throws rather than return nothing when every row filters out", () => {
    expect(() => topicPayload([topicRow({ topico_slug: "inventado" })], SERMONS, TOPICS)).toThrow(
      /none matched/,
    );
  });
});

describe("spotifyPartition", () => {
  const row = (id: string, alive: string) => ({ sermon_id: id, spotify_id: `sp${id}`, alive });

  it("splits the file into live and aged-out sermon ids", () => {
    const { alive, dead } = spotifyPartition([
      row("1", "true"),
      row("2", "false"),
      row("3", "true"),
    ]);
    expect(alive).toEqual(["1", "3"]);
    expect(dead).toEqual(["2"]);
  });

  it("treats anything that is not the literal true as dead", () => {
    // A blank or malformed cell must not read as playable: a dead play button
    // is worse than none.
    const { alive, dead } = spotifyPartition([row("1", ""), row("2", "TRUE"), row("3", "sim")]);
    expect(alive).toEqual([]);
    expect(dead).toEqual(["1", "2", "3"]);
  });

  it("tolerates surrounding whitespace in either column", () => {
    const { alive } = spotifyPartition([{ sermon_id: " 7 ", spotify_id: "x", alive: " true " }]);
    expect(alive).toEqual(["7"]);
  });

  it("returns two empty lists for an empty file", () => {
    expect(spotifyPartition([])).toEqual({ alive: [], dead: [] });
  });
});
