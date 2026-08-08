"""The download completeness check.

Same contract as the other tests here: no network, no ffprobe, no yt-dlp call.
`discard_if_short` is split from the download for exactly that reason -- what it
measures is injected, so the decision is testable on its own.

The bug it exists to stop: SoundCloud serves HLS in ~300 fragments and 403s
under load, and once `fragment_retries` is spent yt-dlp finalises what it has.
The result is a short file that plays, that `already_have()` accepts, and that
every later stage treats as the whole sermon. 46 of 151 downloads came back
that way; nine reached production covering 2-47% of their class.
"""

import json

import config

import fetch


def download(tmp_path, monkeypatch, stem: str, *, declared: float | None,
             measured: float | None):
    """An audio file on disk, its sidecar, and whatever ffprobe would say."""
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    monkeypatch.setattr(config, "AUDIO_DIR", audio)

    if declared is not None:
        (audio / f"{stem}.info.json").write_text(json.dumps({"duration": declared}))

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


def test_the_tolerance_covers_container_rounding(tmp_path, monkeypatch):
    """A healthy download disagrees with SoundCloud by a fraction of a second.
    The real failures were 0.3% to 47%, so the band between is noise."""
    path = download(tmp_path, monkeypatch, "c [3]", declared=3600, measured=3600 * 0.96)
    assert fetch.discard_if_short(path) is True

    short = download(tmp_path, monkeypatch, "d [4]", declared=3600, measured=3600 * 0.94)
    assert fetch.discard_if_short(short) is False


def test_a_track_with_no_sidecar_is_kept(tmp_path, monkeypatch):
    """Nothing to compare against. Refusing here would reject every track whose
    sidecar carries no duration, which is not the failure that was seen."""
    path = download(tmp_path, monkeypatch, "e [5]", declared=None, measured=60)
    assert fetch.discard_if_short(path) is True
    assert path.exists()


def test_a_track_ffprobe_cannot_read_is_kept(tmp_path, monkeypatch):
    path = download(tmp_path, monkeypatch, "f [6]", declared=3600, measured=None)
    assert fetch.discard_if_short(path) is True
    assert path.exists()
