"""Step 2 -- download audio + yt-dlp metadata for everything discover.py listed.

Downloads one track at a time rather than handing yt-dlp the whole channel, so
a single dead track cannot abort the run and a re-run only fetches what is
actually absent. Nothing here writes into the repo: audio is ~170 MB a sermon
and stays in the work directory forever.
"""

import json
import os
import pathlib
import re
import statistics
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


# How much of the sermon has to be in the file. Measured against the audio that
# is really there, a whole download matches its declared duration to within one
# frame, and the smallest real failure -- a single lost 10-second HLS fragment
# out of a 53-minute sermon -- is already 0.3%. The old tolerance was 5%, which
# only looked safe because the measurement could not see a hole: both sides of
# the ratio came out of the same container header.
MIN_DOWNLOADED = 0.999


def content_seconds(stamps: list[float]) -> float | None:
    """Seconds of audio behind a stream's packet timestamps.

    Deliberately not `stamps[-1]`. A file that lost fragments in the middle
    still carries timestamps running to the declared end -- the audio simply
    jumps, and every header keeps reporting the full duration. Counting packets
    and multiplying by the frame length is what notices the ones that are gone.

    The frame length is taken from the stream rather than assumed, because the
    archive holds AAC (1024 samples a frame) and one MP3 (1152); the median
    interval is the frame length by construction, and is immune to the handful
    of gaps that are the thing being measured.
    """
    if len(stamps) < 2:
        # No interval, so no frame length, so no measurement. Treated the same
        # as an unreadable file: discard and fetch it again.
        return None
    frame = statistics.median(b - a for a, b in zip(stamps, stamps[1:]))
    return len(stamps) * frame if frame > 0 else None


def measured_duration(path: pathlib.Path) -> float | None:
    """Seconds of audio in the file, or None when ffprobe cannot read it.

    None means ffprobe ran and rejected the stream, which is a verdict: the
    media is broken. A missing ffprobe raises FileNotFoundError and is left to
    propagate, so a mis-provisioned machine fails loudly instead of deleting
    the archive one track at a time.

    Packet timestamps rather than `format=duration`: that field is the
    container's claim about itself, and comparing it to the sidecar compares a
    claim to the same claim, so a file missing 42 fragments scores a perfect
    100%. The scan costs about half a second on the longest sermon here.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=300, check=True,
        )
        return content_seconds([float(v) for v in out.stdout.split() if v])
    except (subprocess.SubprocessError, ValueError):
        return None


STEM_ID_RE = re.compile(r"^.* \[(?P<id>\d+)\]$")


def forget_row(stem: str) -> None:
    """Drop this sermon's line from rows.jsonl, if it got that far.

    Three stages each keep their own record of what is done, and every one of
    them is keyed by something a re-download does not change. `rows.jsonl` is
    postprocess's, so a sermon left in it is never re-cleaned however many
    times its audio is fetched again -- and the corpus keeps the word count and
    score measured off the damaged file.
    """
    match = STEM_ID_RE.match(stem)
    if not match or not config.ROWS_JSONL.exists():
        return
    kept = [
        line for line in config.ROWS_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip() and str(json.loads(line)["id"]) != match["id"]
    ]
    config.ROWS_JSONL.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")


def discard(path: pathlib.Path) -> None:
    """Remove a download and everything derived from it.

    Deleting the audio alone is only part of "fetch this again". Work is keyed
    by stem, and re-downloading does not change the stem, so `pending_audio()`
    in transcribe.py goes on seeing `raw/<stem>.txt` and calls the sermon done
    -- the track comes back whole and is then skipped forever, still carrying
    the transcript of the truncated version. Clearing the work directory by
    hand was the only cure, and it is what the twelve repaired sermons needed.
    """
    path.unlink(missing_ok=True)
    (config.RAW_DIR / f"{path.stem}.txt").unlink(missing_ok=True)
    (config.ALIGNMENT_DIR / f"{path.stem}.gz").unlink(missing_ok=True)
    forget_row(path.stem)


def discard_if_short(path: pathlib.Path) -> bool:
    """True when the download is whole; discards it and returns False when not.

    SoundCloud serves HLS in ~300 fragments and 403s under load. Once
    `fragment_retries` is spent yt-dlp finalises what it has: a short file that
    plays, that `already_have()` accepts, and that every later stage treats as
    the sermon. 46 of 151 downloads came back this way, nine of them reaching
    production as transcripts covering 2-47% of the class.

    Discarding rather than flagging is deliberate -- it is what makes the next
    run fetch it again, since `already_have()` is the only record of what is
    done.
    """
    measured = measured_duration(path)
    if measured is None:
        # Not "nothing to compare against" -- ffprobe read the file and refused
        # it. An incomplete HLS merge leaves an MP4 whose header is unusable
        # (`trun track id unknown, no tfhd was found`), which fails every
        # transcription attempt forever unless the file goes.
        print(f"  UNREADABLE {path.name} -- discarding for re-download", flush=True)
        discard(path)
        return False

    declared = config.declared_duration(path.stem)
    if declared is None:
        # Here there genuinely is nothing to compare against. Refusing would
        # reject every track whose sidecar carries no duration, which is not
        # the failure that was seen.
        return True

    if measured / declared >= MIN_DOWNLOADED:
        return True

    print(
        f"  SHORT {path.name}: {measured / 60:.1f} of {declared / 60:.1f} min "
        f"({measured / declared:.0%}) -- discarding for re-download",
        flush=True,
    )
    discard(path)
    return False


def download_options() -> dict:
    """yt-dlp's settings for one track.

    A function rather than a constant because `AUDIO_DIR` is monkeypatched in
    tests, and because the one setting that matters most here is worth being
    able to assert on without a network.
    """
    return {
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
        # The default is to skip a fragment that never arrived and write the
        # rest, which is how 36 of 154 downloads ended up with holes in the
        # middle -- 52.8 minutes of preaching, none of it visible to anything
        # downstream, because the container still declared the full duration.
        # Failing the download instead costs a re-fetch; the alternative costs
        # a sermon nobody knows is incomplete.
        "skip_unavailable_fragments": False,
        # SoundCloud hands out opus/m4a already; remuxing would only lose bits
        # and burn CPU that the GPU box needs for transcription.
        "postprocessors": [],
    }


def fetch(track: dict) -> bool:
    try:
        with yt_dlp.YoutubeDL(download_options()) as ydl:
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
