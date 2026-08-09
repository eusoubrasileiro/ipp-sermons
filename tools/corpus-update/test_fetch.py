"""The download completeness check.

Same contract as the other tests here: no network, no ffprobe, no yt-dlp call.
`discard_if_short` is split from the download for exactly that reason -- what it
measures is injected, so the decision is testable on its own. The measurement
itself is split again, into `content_seconds`, which is pure arithmetic over
packet timestamps and needs no media.

The bug it exists to stop: SoundCloud serves HLS in ~300 fragments and 403s
under load, and once `fragment_retries` is spent yt-dlp finalises what it has.
The result is a short file that plays, that `already_have()` accepts, and that
every later stage treats as the whole sermon. 46 of 151 downloads came back
that way; nine reached production covering 2-47% of their class.

That was the *visible* half. The other half is a file that is short in the
middle: yt-dlp skips a fragment it could not fetch and writes the rest, so the
container still declares the full duration and the audio simply jumps. 36 of
154 downloads in the archive were like that, 52.8 minutes of preaching gone,
and every gate passed them -- because the gate compared the header's duration
against the header's duration.
"""

import json

import pytest

import config

import fetch


def download(tmp_path, monkeypatch, stem: str, *, declared: float | None,
             measured: float | None):
    """An audio file on disk, its sidecar, and whatever ffprobe would say."""
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    monkeypatch.setattr(config, "AUDIO_DIR", audio)
    for name in ("RAW_DIR", "ALIGNMENT_DIR"):
        d = tmp_path / name.split("_")[0].lower()
        d.mkdir(exist_ok=True)
        monkeypatch.setattr(config, name, d)

    if declared is not None:
        track_id = stem.rsplit("[", 1)[-1].rstrip("]")
        (audio / f"{stem}.info.json").write_text(json.dumps({
            "duration": declared, "id": int(track_id), "title": stem.rsplit(" [", 1)[0],
            "webpage_url": f"https://soundcloud.com/ipperegrinos/{track_id}",
        }))

    path = audio / f"{stem}.m4a"
    path.write_bytes(b"audio")
    monkeypatch.setattr(fetch, "measured_duration", lambda _p: measured)
    return path


def test_a_whole_download_is_kept(tmp_path, monkeypatch):
    path = download(tmp_path, monkeypatch, "a [1]", declared=3600, measured=3599)
    assert fetch.discard_if_short(path) is True
    assert path.exists()


def test_a_short_download_is_deleted_so_the_next_run_refetches(tmp_path, monkeypatch):
    """Deleting rather than flagging is the point: `already_have()` is the only
    record of what is done, so the file has to go or it is never fetched again."""
    path = download(tmp_path, monkeypatch, "b [2]", declared=64.7 * 60, measured=60)
    assert fetch.discard_if_short(path) is False
    assert not path.exists()
    assert fetch.already_have("2") is False


def test_the_tolerance_is_a_frame_not_a_fragment(tmp_path, monkeypatch):
    """A whole download measures its declared duration to within a frame.

    The old tolerance was 5%, which was honest only because the measurement
    could not see a hole: both sides of the ratio were read out of the same
    container header, so a damaged file scored 100% and the slack was never
    tested. Measured against the audio that is actually there, 5% is 2.6
    minutes of a 53-minute sermon -- and the smallest real failure is one lost
    10-second fragment, which is 0.3%.
    """
    whole = download(tmp_path, monkeypatch, "c [3]", declared=3145, measured=3145.02)
    assert fetch.discard_if_short(whole) is True

    one_fragment_missing = download(tmp_path, monkeypatch, "d [4]",
                                    declared=3145, measured=3145 - 10)
    assert fetch.discard_if_short(one_fragment_missing) is False


def test_a_track_with_no_sidecar_is_kept(tmp_path, monkeypatch):
    """Nothing to compare against. Refusing here would reject every track whose
    sidecar carries no duration, which is not the failure that was seen."""
    path = download(tmp_path, monkeypatch, "e [5]", declared=None, measured=60)
    assert fetch.discard_if_short(path) is True
    assert path.exists()


def test_a_track_ffprobe_cannot_read_is_discarded(tmp_path, monkeypatch):
    """Unreadable is a verdict, not an absence of one.

    Four downloads reached the peer machine so broken that ffmpeg refused the
    header -- `trun track id unknown, no tfhd was found` -- which is what an
    incomplete HLS merge looks like. Keeping them means they fail transcription
    on every run forever, because nothing else ever re-fetches them.

    `measured_duration` returns None only when ffprobe RAN and could not read
    the file; a missing ffprobe raises FileNotFoundError and is not caught.
    """
    path = download(tmp_path, monkeypatch, "f [6]", declared=3600, measured=None)
    assert fetch.discard_if_short(path) is False
    assert not path.exists()


def test_a_track_with_neither_a_sidecar_nor_a_readable_stream_is_discarded(tmp_path, monkeypatch):
    path = download(tmp_path, monkeypatch, "g [7]", declared=None, measured=None)
    assert fetch.discard_if_short(path) is False


def test_discarding_the_audio_also_drops_the_transcript_made_from_it(tmp_path, monkeypatch):
    """Deleting the audio is only half of "fetch this again".

    `transcribe.pending_audio()` calls a stem done when `raw/<stem>.txt` exists,
    and re-downloading does not change the stem. So a discarded track came back
    whole and was then skipped forever, still carrying the transcript of the
    truncated version -- which is why the twelve repaired sermons needed the
    work directory cleaned out by hand before they would transcribe again.
    """
    path = download(tmp_path, monkeypatch, "h [8]", declared=3600, measured=60)
    raw = config.RAW_DIR / "h [8].txt"
    alignment = config.ALIGNMENT_DIR / "h [8].gz"
    raw.write_text("half a sermon")
    alignment.write_bytes(b"gz")

    assert fetch.discard_if_short(path) is False
    assert not raw.exists()
    assert not alignment.exists()


def test_discarding_the_audio_also_drops_the_row_built_from_it(tmp_path, monkeypatch):
    """The third derived artifact, and the one that hides.

    `postprocess` decides what is left to clean from `rows.jsonl`, so a sermon
    left in there is never re-cleaned no matter how many times its audio is
    fetched again -- and `append` would go on carrying the numbers measured off
    the damaged file.
    """
    path = download(tmp_path, monkeypatch, "j [10]", declared=3600, measured=60)
    monkeypatch.setattr(config, "ROWS_JSONL", tmp_path / "rows.jsonl")
    config.ROWS_JSONL.write_text(
        '{"id": 10, "words": 900}\n{"id": 11, "words": 6000}\n', encoding="utf-8"
    )

    assert fetch.discard_if_short(path) is False
    assert config.ROWS_JSONL.read_text(encoding="utf-8") == '{"id": 11, "words": 6000}\n'


def test_discarding_a_track_never_processed_leaves_the_rows_alone(tmp_path, monkeypatch):
    path = download(tmp_path, monkeypatch, "k [11]", declared=3600, measured=60)
    monkeypatch.setattr(config, "ROWS_JSONL", tmp_path / "rows.jsonl")
    config.ROWS_JSONL.write_text('{"id": 12, "words": 6000}\n', encoding="utf-8")

    fetch.discard_if_short(path)

    assert config.ROWS_JSONL.read_text(encoding="utf-8") == '{"id": 12, "words": 6000}\n'


def test_a_whole_download_keeps_its_transcript(tmp_path, monkeypatch):
    """The counterpart, and the one that matters for cost: the sweep runs over
    every downloaded track on every run, so clearing a good sermon's transcript
    would re-transcribe the whole corpus."""
    path = download(tmp_path, monkeypatch, "i [9]", declared=3600, measured=3599)
    raw = config.RAW_DIR / "i [9].txt"
    raw.write_text("a whole sermon")

    assert fetch.discard_if_short(path) is True
    assert raw.exists()


def test_the_sweep_makes_the_fix_retroactive(tmp_path, monkeypatch):
    """`already_have()` accepts any file with an audio suffix, so without the
    sweep `main` never asks for a track it has and the completeness check --
    which only runs after a download -- never sees it. 46 short files sat in
    the work directory exactly that way."""
    whole = download(tmp_path, monkeypatch, "a [1]", declared=3600, measured=3599)
    short = download(tmp_path, monkeypatch, "b [2]", declared=3600, measured=60)

    # One ffprobe answer per file, keyed by name rather than a single constant.
    monkeypatch.setattr(fetch, "measured_duration",
                        lambda p: 3599 if p.name.startswith("a ") else 60)

    assert [t["id"] for t in fetch.sweep_short_downloads()] == ["2"]
    assert whole.exists()
    assert not short.exists()


def test_the_sweep_reaches_a_sermon_the_corpus_already_counts(tmp_path, monkeypatch):
    """The repair, and why the sweep cannot read `pending.json`.

    `discover` lists what SoundCloud has and the corpus does not, so a sermon
    already in metadata.csv is never pending -- which is every one of the 35
    that were live with holes in them. Walking the audio directory instead is
    what lets a damaged download already accounted for be found at all.
    """
    damaged = download(tmp_path, monkeypatch, "counted [7]", declared=3600, measured=3000)

    refetch = fetch.sweep_short_downloads()

    assert not damaged.exists()
    assert [t["url"] for t in refetch] == ["https://soundcloud.com/ipperegrinos/7"]


def test_a_discarded_track_is_named_by_the_sidecar_it_leaves_behind(tmp_path, monkeypatch):
    """`discard()` removes the media and what was derived from it, never the
    `.info.json` -- which is the only thing left that says where to fetch it
    from, since the track is not in `pending.json` either."""
    download(tmp_path, monkeypatch, "gone [8]", declared=3600, measured=60)

    assert fetch.sweep_short_downloads() == [
        {"id": "8", "title": "gone", "url": "https://soundcloud.com/ipperegrinos/8"}
    ]
    assert (config.AUDIO_DIR / "gone [8].info.json").exists()


def test_a_track_without_a_usable_sidecar_is_dropped_not_refetched(tmp_path, monkeypatch):
    """Nothing to fetch from. Discarding is still right -- the file is damaged
    -- but there is no URL, so the next `discover` has to find it."""
    path = download(tmp_path, monkeypatch, "e [5]", declared=None, measured=None)

    assert fetch.sweep_short_downloads() == []
    assert not path.exists()


# --- measuring what is actually in the file ---------------------------------
#
# `27-07-2025 - 2 Reis 10.1-17 [2139294189]` declared 52.42 minutes and held
# 117,364 AAC frames -- 45.42 minutes. 17 gaps, every one an exact multiple of
# ten seconds, 420.0 seconds in total. Ten seconds is the HLS fragment, so 42
# fragments were missing and yt-dlp wrote the file anyway.
#
# Timestamps run to the declared end regardless, which is why the arithmetic
# has to count packets rather than read the last one.

AAC = 1024 / 44100
MP3 = 1152 / 44100

# ffprobe prints pts_time to six decimals, which the fixture reproduces. The
# rounding leaks into the median frame length and scales with packet count: on
# the longest sermon in the archive it is about a tenth of a second in 52
# minutes, four orders of magnitude inside the 0.1% the check allows. Asserting
# to the frame is the honest precision -- demanding more would be testing
# float arithmetic rather than the measurement.
FRAME = 0.03


def stream(frame: float, count: int, *, hole_after: int = 0, hole: float = 0.0) -> list[float]:
    """Packet start times, optionally with `hole` seconds missing partway."""
    stamps, t = [], 0.0
    for i in range(count):
        stamps.append(round(t, 6))
        t += frame + (hole if i + 1 == hole_after else 0.0)
    return stamps


def test_a_whole_stream_measures_its_own_span():
    stamps = stream(AAC, 1000)
    assert fetch.content_seconds(stamps) == pytest.approx(1000 * AAC, abs=FRAME)


def test_a_hole_costs_exactly_what_it_swallowed():
    """The regression. Both streams end at the same timestamp -- the damaged one
    just has fewer packets getting there, and the container reports the end."""
    whole = stream(AAC, 1000)
    damaged = stream(AAC, 1000 - 431, hole_after=200, hole=431 * AAC)

    assert damaged[-1] == pytest.approx(whole[-1], abs=FRAME)
    assert fetch.content_seconds(damaged) == pytest.approx(
        fetch.content_seconds(whole) - 431 * AAC, abs=FRAME
    )


def test_the_frame_length_is_read_from_the_stream_not_assumed():
    """The archive is 154 m4a and one mp3, which pack 1024 and 1152 samples per
    frame. Hard-coding either would misread the other by 12%."""
    assert fetch.content_seconds(stream(MP3, 500)) == pytest.approx(500 * MP3, abs=FRAME)


def test_a_stream_too_short_to_have_a_frame_length_is_no_measurement():
    """One packet gives no interval to take the frame length from. `None` is the
    same verdict as an unreadable file: discard and fetch it again."""
    assert fetch.content_seconds([0.0]) is None
    assert fetch.content_seconds([]) is None


# --- refusing the damage at the source --------------------------------------


def test_a_fragment_yt_dlp_cannot_fetch_aborts_the_download():
    """yt-dlp skips unavailable fragments by default and finalises what it has,
    which is the file with the holes in it. Detection is the second line; this
    is the first, and it is one option."""
    assert fetch.download_options()["skip_unavailable_fragments"] is False
