"""Step 3 -- WhisperX large-v3 on the local GPU, one downloaded sermon at a time.

Two things are deliberate here. First, the ffmpeg pre-filter is copied from
`archive/python-gpu/sermons_ai/transcribex.py`: the existing 455 transcripts were
produced from loudness-normalised, denoised, high-passed 16 kHz mono, and feeding
raw SoundCloud audio instead would shift both the text and the alignment
confidence that becomes the `score` column.

Second, transcription is delegated to the box's existing WhisperX install rather
than reimplemented. It already pins large-v3 / float16 / pt against this GPU and
emits the same word-level alignment .gz that the score function reads.

Work is keyed by the yt-dlp audio stem (`Title [id]`), which is unique per track,
so an interrupted run resumes by simply skipping stems that already have output.
"""

import argparse
import gzip
import json
import os
import pathlib
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import config


def shard(items: list, index: int, count: int) -> list:
    """The `index`-th of `count` contiguous slices, 1-based.

    Contiguous rather than round-robin, and 1-based on purpose. The backlog is
    walked in sorted order, so a box that is already running is partway through
    the head of it; giving shard 1 the head and the newcomer the tail means the
    two never meet, and neither has to be restarted to join. Round-robin would
    interleave them and send the joining box straight to sermons the incumbent
    is about to reach.
    """
    if count < 1 or not 1 <= index <= count:
        raise ValueError(f"shard {index}/{count} is not a slice of anything")
    size, rest = divmod(len(items), count)
    # The first `rest` slices take one extra, so no item is dropped when the
    # backlog does not divide evenly.
    start = (index - 1) * size + min(index - 1, rest)
    return items[start : start + size + (1 if index <= rest else 0)]


def cuda_env() -> dict[str, str]:
    """CTranslate2 dlopens libcudnn/libcublas by soname and does not look inside
    the venv, so WhisperX falls back to a CPU path (or dies) unless the pip-
    installed NVIDIA libs are on the loader path. Torch gets away with it
    because it links them at build time; ctranslate2 does not."""
    site = config.ML_TOOLS_DIR / ".venv/lib/python3.12/site-packages/nvidia"
    libs = [str(site / pkg / "lib") for pkg in ("cudnn", "cublas")]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join([*libs, env.get("LD_LIBRARY_PATH", "")]).rstrip(":")
    return env


def to_wav(audio: pathlib.Path) -> pathlib.Path | None:
    wav = config.WAV_DIR / f"{audio.stem}.wav"
    if wav.exists():
        return wav
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(audio),
        "-vn",
        "-af",
        config.FFMPEG_FILTERS,
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not wav.exists():
        print(f"  ffmpeg failed for {audio.name}: {proc.stderr[-400:]}", file=sys.stderr, flush=True)
        return None
    return wav


def coverage(gz: pathlib.Path, wav: pathlib.Path) -> float:
    """Fraction of the sermon the transcript actually reaches.

    WhisperX can come back having transcribed only the first few minutes -- the
    VAD drops the rest -- and because alignment then succeeds on those few
    segments, transcribe.py writes a perfectly well-formed .txt and .gz and
    exits 0. Nothing downstream would notice: a fifth of a sermon scores fine
    and indexes fine. Length is the only thing that gives it away.

    The denominator is what SoundCloud says the sermon lasts, NOT the WAV.
    Measuring the WAV makes this guard circular: hand it a truncated download
    and WhisperX transcribes all of it, so `segments[-1].end / duration` is 1.0
    and a 1-minute file passes as a complete 64.7-minute class.

    Falls back to measuring the WAV only when there is no sidecar to ask -- a
    peer transcribing handed-over audio may not have one, and refusing there
    would stall the backlog over a file that is probably fine.
    """
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        segments = json.load(f).get("segments", [])
    if not segments:
        return 0.0

    # 16 kHz, 16-bit, mono, minus the 44-byte RIFF header.
    duration = config.declared_duration(wav.stem) or (wav.stat().st_size - 44) / (16000 * 2)
    return segments[-1]["end"] / duration if duration > 0 else 0.0


# Sermons end with prayer and often a stretch of music or silence, so demanding
# the full duration would reject good runs; a fifth of a sermon is the failure
# actually seen.
MIN_COVERAGE = 0.90

# Set after the first successful sermon reports which compute type and card
# produced it, so the run says so exactly once.
REPORTED_COMPUTE_TYPE = False


def provenance(stdout: str) -> str | None:
    """The worker's compute-type line, pulled out of its captured output.

    The worker prints which compute type and which card produced the
    transcript, but its stdout is a pipe this module reads only when something
    failed. Since a peer machine may be transcribing at a different compute
    type than the corpus baseline, that line is the only record of which path
    a given transcript came from, and it has to reach the log on success too.
    """
    for line in stdout.splitlines():
        if line.startswith("compute_type="):
            return line.strip()
    return None


def transcribe(wav: pathlib.Path, compute_type: str | None = None) -> bool:
    """Returns True when a complete transcript and its alignment landed.

    Judged by the output files rather than the exit code, and retried at a
    smaller batch size first, because the common failure on a 6 GB card is a
    CUDA OOM whose only cure is less batch.
    """
    txt = config.RAW_DIR / f"{wav.stem}.txt"
    gz = config.RAW_DIR / f"{wav.stem}.gz"

    for batch in config.TRANSCRIBE_BATCH_SIZES:
        cmd = [
            str(config.TRANSCRIBE_PYTHON),
            str(config.TRANSCRIBE_SCRIPT),
            "--wav",
            str(wav),
            "--out-dir",
            str(config.RAW_DIR),
            "--batch-size",
            str(batch),
            *(["--compute-type", compute_type] if compute_type else []),
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=cuda_env()
        )

        if txt.exists() and gz.exists() and txt.stat().st_size > 0:
            got = coverage(gz, wav)
            if got >= MIN_COVERAGE:
                global REPORTED_COMPUTE_TYPE
                # Once per run, not per sermon: it cannot change mid-run, and a
                # line per sermon would bury the [n/m] progress it sits beside.
                if not REPORTED_COMPUTE_TYPE and (line := provenance(proc.stdout)):
                    print(f"  {line}", flush=True)
                    REPORTED_COMPUTE_TYPE = True
                # Keep the alignment beside the transcripts, where the score
                # function inherited from doc_preproc.py expects to find it.
                shutil.move(str(gz), str(config.ALIGNMENT_DIR / gz.name))
                return True
            reason = f"only reached {got:.0%} of the audio"
        else:
            errors = [ln for ln in proc.stdout.splitlines() if "rror" in ln]
            reason = errors[-1] if errors else "no output"

        # Leave nothing behind, or the next run treats the stem as done.
        txt.unlink(missing_ok=True)
        gz.unlink(missing_ok=True)
        print(f"  batch {batch} failed: {reason}", flush=True)

    return False


def discard_incomplete() -> list[str]:
    """Deletes raw/alignment pairs that do not cover their sermon. Returns them.

    `coverage()` only ever runs inside `transcribe()`, which is the local path.
    Transcripts arriving from a peer through `peer.sh collect` bypass it, and
    the peer cannot check for itself: `dispatch` ships audio without the
    `.info.json` that says how long the sermon is, so `coverage` there falls
    back to measuring the .wav and truncated audio validates against itself.

    This is the same test applied where every transcript passes regardless of
    which machine produced it. It walks the whole alignment directory rather
    than what a collect just brought home, because a locally produced
    transcript has already passed exactly this check and re-running it is a
    gzip read -- cheap enough that having one answer for the whole directory is
    worth more than narrowing it.

    Deleting rather than quarantining keeps one meaning for "a raw transcript
    exists": `pending_audio()` reads that directory to decide what is left to
    do, so a file kept aside would silently mark the sermon done.
    """
    removed = []
    for gz in sorted(config.ALIGNMENT_DIR.glob("*.gz")):
        # No sidecar means no sermon length to judge against, and the .wav that
        # `coverage` would fall back to was deleted once transcription finished.
        if config.declared_duration(gz.stem) is None:
            continue
        if coverage(gz, config.WAV_DIR / f"{gz.stem}.wav") >= MIN_COVERAGE:
            continue
        gz.unlink(missing_ok=True)
        (config.RAW_DIR / f"{gz.stem}.txt").unlink(missing_ok=True)
        removed.append(gz.stem)
    return removed


def assigned_elsewhere() -> set[str]:
    """Audio file names another machine is already working on.

    Empty, and cheap, when no peer has ever run -- the single-machine case has
    to behave exactly as it did before peers existed.
    """
    if not config.ASSIGNED_DIR.is_dir():
        return set()
    return {
        line.strip()
        for handout in config.ASSIGNED_DIR.glob("*.txt")
        for line in handout.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def pending_audio() -> list[pathlib.Path]:
    done = {p.stem for p in config.RAW_DIR.glob("*.txt")}
    taken = assigned_elsewhere()
    return sorted(
        p
        for p in config.AUDIO_DIR.iterdir()
        if p.suffix.lower() in config.AUDIO_SUFFIXES
        and p.stem not in done
        and p.name not in taken
    )


def parse_shard(spec: str) -> tuple[int, int]:
    try:
        index, count = (int(part) for part in spec.split("/", 1))
    except ValueError:
        raise SystemExit(f"--shard wants i/n, got {spec!r}") from None
    return index, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        default="1/1",
        help="i/n -- take only the i-th of n contiguous slices of the backlog, "
        "so a second machine can work the same corpus without collisions",
    )
    parser.add_argument(
        "--compute-type",
        default=os.environ.get("IPP_COMPUTE_TYPE") or None,
        help="force a compute type; omit to let each GPU use the best it has",
    )
    args = parser.parse_args()
    index, count = parse_shard(args.shard)

    config.ensure_dirs()
    todo = pending_audio()
    if count > 1:
        whole = len(todo)
        todo = shard(todo, index, count)
        print(f"shard {index}/{count}: {len(todo)} of {whole} pending", flush=True)
    print(f"to transcribe: {len(todo)}", flush=True)
    if not todo:
        return

    ok = 0
    # The ffmpeg pre-filter costs several CPU-minutes a sermon and the GPU sits
    # idle through all of it. Converting the next sermon while the current one
    # transcribes hides that entirely, at the cost of one extra wav on disk --
    # worth it over a backlog measured in days.
    with ThreadPoolExecutor(max_workers=1) as pre:
        upcoming = pre.submit(to_wav, todo[0])

        for i, audio in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {audio.stem}", flush=True)
            wav = upcoming.result()
            upcoming = pre.submit(to_wav, todo[i]) if i < len(todo) else None

            if wav is None:
                continue
            if transcribe(wav, args.compute_type):
                ok += 1
            else:
                print(f"  TRANSCRIPTION FAILED {audio.stem}", flush=True)
            # ~115 MB of 16 kHz mono per sermon-hour, and nothing downstream
            # reads it again; the audio is still on disk if a redo is needed.
            wav.unlink(missing_ok=True)

    print(f"transcribed {ok}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
