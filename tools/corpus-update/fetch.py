"""Step 2 -- download audio + yt-dlp metadata for everything discover.py listed.

Downloads one track at a time rather than handing yt-dlp the whole channel, so
a single dead track cannot abort the run and a re-run only fetches what is
actually absent. Nothing here writes into the repo: audio is ~170 MB a sermon
and stays in the work directory forever.
"""

import json
import os
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

import config

WORKERS = int(os.environ.get("IPP_FETCH_WORKERS", "2"))


# Lives in config so that transcribe.py can find audio without importing this
# module, which would drag yt-dlp onto a machine that only ever transcribes.
AUDIO_SUFFIXES = config.AUDIO_SUFFIXES


def audio_path(track_id: str) -> "pathlib.Path | None":
    """The finished audio file for a track, if there is one.

    yt-dlp writes the .info.json *before* the media, and leaves `.part`/`.ytdl`
    scratch files behind mid-download, so presence of "some file with this id"
    proves nothing. Only a real audio suffix counts as done.
    """
    for p in config.AUDIO_DIR.glob(f"*[[]{track_id}[]].*"):
        if p.suffix.lower() in AUDIO_SUFFIXES:
            return p
    return None


def already_have(track_id: str) -> bool:
    return audio_path(track_id) is not None


# How much of the sermon has to be in the file. SoundCloud's own duration and
# the container's disagree by a fraction of a second on a healthy download; the
# failures were 0.3% to 47%, so anything in between is noise.
MIN_DOWNLOADED = 0.95


def measured_duration(path: pathlib.Path) -> float | None:
    """Seconds of audio actually in the file, or None if ffprobe cannot say."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def discard_if_short(path: pathlib.Path) -> bool:
    """True when the download is whole; deletes it and returns False when not.

    SoundCloud serves HLS in ~300 fragments and 403s under load. Once
    `fragment_retries` is spent yt-dlp finalises what it has: a short file that
    plays, that `already_have()` accepts, and that every later stage treats as
    the sermon. 46 of 151 downloads came back this way, nine of them reaching
    production as transcripts covering 2-47% of the class.

    Deleting rather than flagging is deliberate -- it is what makes the next run
    fetch it again, since `already_have()` is the only record of what is done.
    """
    declared = config.declared_duration(path.stem)
    measured = measured_duration(path)
    if declared is None or measured is None:
        # Nothing to compare against. Refusing here would reject every track
        # whose sidecar carries no duration, which is not the failure we saw.
        return True

    if measured / declared >= MIN_DOWNLOADED:
        return True

    print(
        f"  SHORT {path.name}: {measured / 60:.1f} of {declared / 60:.1f} min "
        f"({measured / declared:.0%}) -- discarding for re-download",
        flush=True,
    )
    path.unlink(missing_ok=True)
    return False


def fetch(track: dict) -> bool:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(config.AUDIO_DIR / config.AUDIO_OUTPUT_TEMPLATE),
        "writeinfojson": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # SoundCloud serves HLS in ~300 two-second fragments; fetched one at a
        # time a sermon takes minutes, almost all of it round-trip latency. But
        # push this (or WORKERS) up and it starts answering 403 to the fragment
        # manifests -- 4 x 4 is the setting that got through the backlog.
        "concurrent_fragment_downloads": 4,
        # 403s from throttling are transient, and a track abandoned mid-backlog
        # is a hole someone has to notice; let yt-dlp back off and retry first.
        "retries": 10,
        "fragment_retries": 10,
        "retry_sleep_functions": {"http": lambda n: min(2**n, 60)},
        # SoundCloud hands out opus/m4a already; remuxing would only lose bits
        # and burn CPU that the GPU box needs for transcription.
        "postprocessors": [],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([track["url"]])
        path = audio_path(track["id"])
        return path is not None and discard_if_short(path)
    except Exception as exc:  # a private or deleted track should not stop the backlog
        print(f"  FAILED {track['id']} {track['title']!r}: {exc}", flush=True)
        return False


def sweep_short_downloads(pending: list[dict]) -> int:
    """Re-checks what previous runs left behind, discarding anything short.

    Without this the fix is not retroactive: `already_have()` accepts any file
    with an audio suffix, so `main` never asks for a track it has, and the
    completeness check in `fetch` -- which only runs after a download -- never
    sees it. 46 short files sat in the work directory that way, and the only
    thing that would have re-fetched them was somebody deleting them by hand.

    Costs one ffprobe per downloaded track per run, which is seconds over the
    whole archive and buys a pipeline that repairs itself.
    """
    discarded = 0
    for track in pending:
        path = audio_path(track["id"])
        if path is not None and not discard_if_short(path):
            discarded += 1
    return discarded


def main() -> None:
    config.ensure_dirs()
    pending = json.loads(config.PENDING_JSON.read_text(encoding="utf-8"))

    discarded = sweep_short_downloads(pending)
    if discarded:
        print(f"discarded {discarded} short download(s) from a previous run", flush=True)

    todo = [t for t in pending if not already_have(t["id"])]
    print(f"pending {len(pending)}, to download {len(todo)}", flush=True)

    done = 0
    ok = 0
    # Network-bound and independent per track, so overlap them -- but modestly,
    # since transcription is running on the same box and SoundCloud throttles.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch, t): t for t in todo}
        for fut in as_completed(futures):
            track = futures[fut]
            done += 1
            ok += bool(fut.result())
            print(f"[{done}/{len(todo)}] {track['id']} {track['title']}", flush=True)

    print(f"downloaded {ok}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
