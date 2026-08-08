import type { TranscriptResponse } from "@ipp/shared";
import { describe, expect, it } from "vitest";
import { sermonPage } from "../../src/lib/seo/sermon-page.ts";

/**
 * What a crawler — and a WhatsApp link preview — actually reads.
 *
 * The heading and the date have to be written the way the hydrated page writes
 * them (`@ipp/shared`'s `stripLeadingDate`/`formatDate`), because these are the
 * same page a few hundred milliseconds apart. A divergence here shows up as the
 * article visibly changing under the reader.
 */

const transcript = (over: Partial<TranscriptResponse> = {}): TranscriptResponse => ({
  id: "2123517330",
  title: "01-01-2023 - Tito 2",
  artist: "Reverendo Bruno Melo",
  date: "2023-01-01",
  durationStr: "1:02:39",
  soundcloudUrl: "https://soundcloud.com/ipperegrinos/tito-2",
  spotifyUrl: "https://open.spotify.com/episode/4rOoJ6Egrf8K2IrywzwOMk",
  words: 6424,
  text: "Como sempre, é um motivo de alegria estar aqui diante dessa querida igreja.",
  ...over,
});

describe("sermonPage", () => {
  it("titles the page with the sermon, not with the corpus's date prefix", () => {
    const page = sermonPage(transcript());
    expect(page.title).toBe("Tito 2 — Reverendo Bruno Melo");
    expect(page.path).toBe("/sermao/2123517330");
    expect(page.ogType).toBe("article");
  });

  it("describes the sermon with its own opening words, not with boilerplate", () => {
    const page = sermonPage(transcript());
    expect(page.description).toContain("Como sempre, é um motivo de alegria");
  });

  it("puts the transcript in the body, so the page has something to index", () => {
    const page = sermonPage(transcript());
    expect(page.body).toContain("motivo de alegria estar aqui diante dessa querida igreja");
    expect(page.body).toContain("<h1");
    expect(page.body).toContain("Reverendo Bruno Melo");
  });

  it("dates the article in the pt-BR form the hydrated header uses", () => {
    const page = sermonPage(transcript());
    expect(page.body).toContain('<time datetime="2023-01-01">01 jan 2023</time>');
  });

  it("links the audio, which is the thing the archive actually hosts", () => {
    const page = sermonPage(transcript());
    expect(page.body).toContain('href="https://soundcloud.com/ipperegrinos/tito-2"');
    expect(page.body).toContain('href="https://open.spotify.com/episode/4rOoJ6Egrf8K2IrywzwOMk"');
  });

  it("omits a play link the corpus does not have rather than linking nowhere", () => {
    const page = sermonPage(transcript({ soundcloudUrl: null, spotifyUrl: null }));
    expect(page.body).not.toContain("SoundCloud");
    expect(page.body).not.toContain("Spotify");
  });

  it("escapes a transcript that contains markup", () => {
    // The corpus is transcribed speech rebuilt from a CSV, and a preacher
    // reading an email address or a verse reference has already produced both
    // `<` and `&` in this archive.
    const page = sermonPage(
      transcript({
        title: `<script>alert("t")</script>`,
        text: `Ele disse <b>isto</b> & aquilo.`,
      }),
    );

    expect(page.body).not.toContain("<script>");
    expect(page.body).not.toContain("<b>isto</b>");
    expect(page.body).toContain("&lt;b&gt;isto&lt;/b&gt; &amp; aquilo.");
  });

  it("says the text is machine-transcribed, as the reading page does", () => {
    // Not decoration: a preacher misquoted by a transcription error is a real
    // problem for the church, and a crawler's snippet may be all anyone reads.
    expect(sermonPage(transcript()).body).toContain("Transcrição automática");
  });

  it("carries the site navigation, so one sermon is not a dead end", () => {
    // A crawler that lands on a sermon from a search result must be able to
    // reach the rest of the archive from there, without JavaScript and without
    // having to fetch and trust sitemap.xml.
    const body = sermonPage(transcript()).body;
    for (const href of ["/temas", "/biblia", "/series", "/pregadores", "/datas"]) {
      expect(body).toContain(`href="${href}"`);
    }
  });

  it("still renders a sermon whose transcript came back empty", () => {
    const page = sermonPage(transcript({ text: "   " }));
    expect(page.body).toContain("<h1");
    expect(page.description.length).toBeGreaterThan(0);
  });
});
