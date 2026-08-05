"""Step 2 -- download audio + yt-dlp metadata for everything discover.py listed.

Downloads one track at a time rather than handing yt-dlp the whole channel, so
a single dead track cannot abort the run and a re-run only fetches what is
actually absent. Nothing here writes into the repo: audio is ~170 MB a sermon
and stays in the work directory forever.
"""

import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

import config

WORKERS = int(os.environ.get("IPP_FETCH_WORKERS", "2"))


AUDIO_SUFFIXES = {".opus", ".m4a", ".mp3", ".ogg", ".wav", ".aac", ".webm", ".flac"}


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
        return already_have(track["id"])
    except Exception as exc:  # a private or deleted track should not stop the backlog
        print(f"  FAILED {track['id']} {track['title']!r}: {exc}", flush=True)
        return False


def main() -> None:
    config.ensure_dirs()
    pending = json.loads(config.PENDING_JSON.read_text(encoding="utf-8"))

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
