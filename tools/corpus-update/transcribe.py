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

import gzip
import json
import os
import pathlib
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import config
import fetch


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
    """Fraction of the audio the transcript actually reaches.

    WhisperX can come back having transcribed only the first few minutes -- the
    VAD drops the rest -- and because alignment then succeeds on those few
    segments, transcreve.py writes a perfectly well-formed .txt and .gz and
    exits 0. Nothing downstream would notice: a fifth of a sermon scores fine
    and indexes fine. Length is the only thing that gives it away.
    """
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        segments = json.load(f).get("segments", [])
    if not segments:
        return 0.0
    # 16 kHz, 16-bit, mono, minus the 44-byte RIFF header.
    duration = (wav.stat().st_size - 44) / (16000 * 2)
    return segments[-1]["end"] / duration if duration > 0 else 0.0


# Sermons end with prayer and often a stretch of music or silence, so demanding
# the full duration would reject good runs; a fifth of a sermon is the failure
# actually seen.
MIN_COVERAGE = 0.90


def transcribe(wav: pathlib.Path) -> bool:
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
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=cuda_env()
        )

        if txt.exists() and gz.exists() and txt.stat().st_size > 0:
            got = coverage(gz, wav)
            if got >= MIN_COVERAGE:
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


def pending_audio() -> list[pathlib.Path]:
    done = {p.stem for p in config.RAW_DIR.glob("*.txt")}
    return sorted(
        p
        for p in config.AUDIO_DIR.iterdir()
        if p.suffix.lower() in fetch.AUDIO_SUFFIXES and p.stem not in done
    )


def main() -> None:
    config.ensure_dirs()
    todo = pending_audio()
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
            if transcribe(wav):
                ok += 1
            else:
                print(f"  TRANSCRIPTION FAILED {audio.stem}", flush=True)
            # ~115 MB of 16 kHz mono per sermon-hour, and nothing downstream
            # reads it again; the audio is still on disk if a redo is needed.
            wav.unlink(missing_ok=True)

    print(f"transcribed {ok}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
