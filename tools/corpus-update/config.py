"""Shared paths and constants for the corpus update pipeline.

Values here are inherited from the retired GPU pipeline (`archive/python-gpu`),
not chosen: the SoundCloud channel, the ffmpeg filter chain, the WhisperX model
and the score cutoff all have to match, or new rows stop being comparable to the
455 already in `data/metadata.csv`.
"""

import json
import os
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
METADATA_CSV = DATA_DIR / "metadata.csv"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
PREACHER_NAMES_CSV = DATA_DIR / "preacher_names.txt"

# Audio is ~170 MB per sermon and the existing 77 GB deliberately lives outside
# the repo and off the VPS. Everything the pipeline downloads or generates from
# audio stays under here; only .txt transcripts and metadata.csv cross back.
WORK_DIR = pathlib.Path(os.environ.get("IPP_WORK_DIR", "/mnt/Data/ipp-sermons-work"))
AUDIO_DIR = WORK_DIR / "audio"
WAV_DIR = WORK_DIR / "wav"
RAW_DIR = WORK_DIR / "raw"
ALIGNMENT_DIR = WORK_DIR / "alignment"
PENDING_JSON = WORK_DIR / "pending.json"
ROWS_JSONL = WORK_DIR / "rows.jsonl"
# One file per peer machine, listing the audio handed to it. Read by the
# transcribe stage so that this box leaves those sermons alone; written and
# removed by peer.sh. A positional split could not do this job: the backlog
# shrinks from the head as work completes, so "the second half" slides
# backwards over sermons a peer took hours ago.
ASSIGNED_DIR = WORK_DIR / "assigned"

SOUNDCLOUD_USER_URL = "https://soundcloud.com/ipperegrinos"
SOUNDCLOUD_TRACKS_URL = f"{SOUNDCLOUD_USER_URL}/tracks"
SPOTIFY_SHOW_ID = "1DgxzkzYvNGLNv7UawbEUP"

# yt-dlp's own template, kept because the `audio` column of every existing row
# was written by it and metadata.py rediscovers files by globbing on the id.
AUDIO_OUTPUT_TEMPLATE = "%(title)s [%(id)s].%(ext)s"

# What counts as a downloaded sermon. Here rather than in fetch.py because the
# transcribe stage needs it too, and a machine that only transcribes should not
# have to install yt-dlp to import the constant.
AUDIO_SUFFIXES = {".opus", ".m4a", ".mp3", ".ogg", ".wav", ".aac", ".webm", ".flac"}

# Every file the pipeline makes is named for the yt-dlp stem `<title> [<id>]`,
# and every stage that has to get back from a file to a sermon parses it. One
# definition, for the same reason as AUDIO_SUFFIXES: fetch keys `rows.jsonl` on
# the id and postprocess keys the corpus on the title, and if the two ever
# disagree about what a stem looks like, a discarded download keeps its row.
STEM_ID_RE = re.compile(r"^(?P<title>.*) \[(?P<id>\d+)\]$")


def info_json_for(stem: str) -> dict | None:
    """yt-dlp's sidecar for a downloaded track, or None if it never landed.

    Here for the same reason as AUDIO_SUFFIXES: fetch, transcribe and
    postprocess all need it, and transcribe must not import fetch.

    `duration` in this file is what SoundCloud says the sermon is, which makes
    it the only trustworthy denominator in the pipeline. Every other number
    available -- the container header, the WAV's byte length -- is derived from
    the download, so measuring against one of those asks a damaged file whether
    it is damaged and it always answers no.
    """
    path = AUDIO_DIR / f"{stem}.info.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def declared_duration(stem: str) -> float | None:
    """Seconds of sermon according to SoundCloud, or None when unknowable."""
    info = info_json_for(stem)
    duration = float((info or {}).get("duration") or 0)
    return duration if duration > 0 else None

# The transcriber runs under the ml-tools venv, which already has WhisperX
# large-v3 + torch/cu118 working against the box's GTX 1660 SUPER. Only the
# interpreter is borrowed -- the worker itself is ours (see whisperx_worker.py),
# because that tool's VAD choice is not the one that produced this corpus.
ML_TOOLS_DIR = pathlib.Path.home() / "Projects/amiticia/repositories/tools/ml-tools/audio"
TRANSCRIBE_PYTHON = ML_TOOLS_DIR / ".venv/bin/python"
TRANSCRIBE_SCRIPT = pathlib.Path(__file__).parent / "whisperx_worker.py"
# One entry per attempt, in order. The card has 6 GB and the desktop
# (gnome/chrome/obs) permanently holds ~1 GB of it, leaving too little for
# large-v3 float16 at the batch size 4 the old headless setup used -- it OOMs.
# 2 fits. The repeat at 2 is not a typo: the other failure mode, the VAD losing
# the back half of a sermon, is intermittent, so a plain retry usually clears it;
# 1 is the last resort for when the desktop is genuinely using more than usual.
TRANSCRIBE_BATCH_SIZES = [2, 2, 1]

# Loudness normalisation, spectral denoise and a high-pass to kill room rumble,
# then Whisper's required 16 kHz mono. Straight from transcribex.py.
FFMPEG_FILTERS = "loudnorm,afftdn=nf=-25,highpass=f=150"

LANGUAGE_TOOL_RULES = [
    "UPPERCASE_AFTER_COMMA",
    "UPPERCASE_SENTENCE_START",
    "VERB_COMMA_CONJUNCTION",
    "ALTERNATIVE_CONJUNCTIONS_COMMA",
    "PORTUGUESE_WORD_REPEAT_RULE",
]

SPACY_MODEL = "pt_core_news_lg"

# language_tool_python drives a bundled Java LanguageTool server, and this box
# has no system JVM. setup.sh unpacks a private Temurin JRE here rather than
# asking for a sudo apt install; when it is absent the cleaner says so and
# carries on without the grammar passes.
VENV_DIR = pathlib.Path(os.environ.get("IPP_VENV_DIR", "/mnt/Data/venv/ipp-sermons"))
JRE_DIR = VENV_DIR / "jre"

# Mirrors MIN_SCORE in backend/src/lib/corpus.ts. Only used for reporting here;
# low-scoring rows are still written, and the loader is what filters them.
MIN_SCORE = 50


def ensure_dirs() -> None:
    for d in (AUDIO_DIR, WAV_DIR, RAW_DIR, ALIGNMENT_DIR, ASSIGNED_DIR):
        d.mkdir(parents=True, exist_ok=True)
