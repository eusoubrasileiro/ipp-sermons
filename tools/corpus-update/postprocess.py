"""Step 4 -- turn a raw transcript into a cleaned corpus file plus one CSV row.

Every column is derived the way `archive/python-gpu` derived it, so a row added
today is indistinguishable from one added in 2024. Results are appended to
rows.jsonl one at a time, which is what makes the stage resumable: the cleaner
takes ~1 min a sermon and losing an hour of it to a crash at the end would be
avoidable waste.
"""

import datetime
import json
import pathlib
import re

import pandas as pd
from rapidfuzz import fuzz, process as fuzzy

import clean
import config
import spotify_ids

# Titles are `<title> [<soundcloud id>]` on disk; the corpus stores the title.
STEM_ID_RE = re.compile(r"^(?P<title>.*) \[(?P<id>\d+)\]$")

DATE_IN_TITLE_RE = re.compile(r"(\d{2}).+(\d{2}).+(\d{4})")


def resolve_date(text: str, timestamp: int) -> datetime.date:
    """The sermon's date, from its title when the title is trustworthy.

    Titles carry a `DD-MM-YYYY` prefix by convention, but only by convention:
    the corpus contains "05-01-20245", "0223-05-07" and a run of episodes with
    no date at all. The original took whatever the regex produced and left the
    column blank when it did not match, which is where the six unusable dates in
    metadata.csv came from. Here the publication timestamp -- always present,
    always sane -- backstops both cases, and the plausibility window matches
    `resolveDate` in backend/src/lib/corpus.ts so the loader never has to.
    """
    published = datetime.datetime.fromtimestamp(timestamp, datetime.UTC).date()

    match = DATE_IN_TITLE_RE.search(text)
    if match:
        try:
            d, m, y = map(int, match.groups())
            parsed = datetime.date(y, m, d)
            if 2015 <= parsed.year <= 2030:
                return parsed
        except ValueError:
            pass

    return published


def preacher_matcher():
    df = pd.read_csv(config.PREACHER_NAMES_CSV)
    return df.name.str.lower().to_list(), df.full_name.to_list()


def artist_candidates(info: dict) -> list[str]:
    """Where the preacher's name might be, best source first.

    The original read yt-dlp's `artist` field, and for the 2024-and-earlier
    corpus that worked. It does not any more: SoundCloud now returns the channel
    ("Igreja Presbiteriana Peregrinos") there for essentially every track. The
    name survives in the description, which by house style ends "<topic> por
    <Preacher>". The old field is still tried second, for tracks that predate
    the change.
    """
    candidates = []

    description = (info.get("description") or "").strip()
    if " por " in description:
        candidates.append(description.rsplit(" por ", 1)[-1].splitlines()[0])

    if info.get("artist"):
        candidates.append(info["artist"])

    return candidates


def resolve_artist(info: dict, preachers: list[str], full_names: list[str]) -> str | None:
    """Map a free-text preacher reference onto the canonical name list.

    References are inconsistent ("Pr.", "Rev.", "Presb.", plain first names), so
    strip the honorific and fuzzy-match the remainder. Cutoff 70 is the
    original's; below that the column is left blank rather than guessed, and the
    indexer substitutes "Desconhecido".
    """
    for raw in artist_candidates(info):
        name = str(raw).lower()
        for prefix in [
            "seminarista", "presb.", "pr.", "rev.", "pb.", "presb",
            "rev", "semi.", "pres.", "preb.", "pastor", "reverendo",
        ]:
            name = name.replace(prefix, "")

        match = fuzzy.extractOne(name, preachers, scorer=fuzz.ratio, score_cutoff=70)
        if match:
            return full_names[int(match[2])]

    return None


def info_json_for(stem: str) -> dict | None:
    path = config.AUDIO_DIR / f"{stem}.info.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def audio_file_for(stem: str) -> str | None:
    for p in config.AUDIO_DIR.glob(f"{glob_escape(stem)}.*"):
        if p.suffix.lower() in {".opus", ".m4a", ".mp3", ".ogg", ".wav", ".aac", ".webm", ".flac"}:
            return p.name
    return None


def glob_escape(s: str) -> str:
    """Sermon titles contain `[`, `]` and `?`, all of which are glob syntax."""
    return re.sub(r"([\[\]?*])", r"[\1]", s)


def done_ids() -> set[str]:
    """The soundcloud ids already turned into rows, as strings.

    `str()` is load-bearing. `build_row` writes the id as an int, and must keep
    doing so -- append.py's DTYPES are what stop a new CSV line from formatting
    differently to the 455 already in metadata.csv. The comparison side is the
    regex capture in `pending`, which is always a string, so the conversion has
    to happen here. It did not, and `"111" not in {111}` is always true: every
    re-run re-cleaned every raw transcript and rewrote every corpus .txt.
    """
    if not config.ROWS_JSONL.exists():
        return set()
    with config.ROWS_JSONL.open(encoding="utf-8") as f:
        return {str(json.loads(line)["id"]) for line in f if line.strip()}


def pending(raw_paths: list[pathlib.Path], have: set[str]) -> list[pathlib.Path]:
    """The raw transcripts still to clean, in the order given."""
    return [p for p in raw_paths if (m := STEM_ID_RE.match(p.stem)) and m["id"] not in have]


def build_row(raw_path: pathlib.Path, preachers, full_names, spotify) -> dict | None:
    match = STEM_ID_RE.match(raw_path.stem)
    if not match:
        print(f"  skipping {raw_path.stem}: no soundcloud id in filename", flush=True)
        return None
    title, track_id = match["title"], match["id"]

    info = info_json_for(raw_path.stem)
    if info is None:
        print(f"  skipping {raw_path.stem}: no .info.json", flush=True)
        return None

    alignment = config.ALIGNMENT_DIR / f"{raw_path.stem}.gz"
    if not alignment.exists():
        print(f"  skipping {raw_path.stem}: no alignment .gz, cannot score", flush=True)
        return None

    # The corpus keys transcripts by title alone; the id lives in the CSV.
    transcript_name = f"{title}.txt"
    words, sentences, sent_ratio = clean.process(
        raw_path, config.TRANSCRIPTS_DIR / transcript_name
    )

    duration = float(info["duration"])
    timestamp = int(info["timestamp"])
    description = info.get("description") or ""

    return {
        "name": title,
        "wav": True,
        "transcribed": True,
        "processed": True,
        "description": description,
        "audio": audio_file_for(raw_path.stem) or "",
        "txt": transcript_name,
        "artist": resolve_artist(info, preachers, full_names),
        "duration_str": info.get("duration_string") or "",
        "id": int(track_id),
        "view_count": info.get("view_count") or 0,
        "duration": duration,
        "timestamp": timestamp,
        "sc_suffix_url": info.get("webpage_url_basename") or "",
        "sp_suffix_url": spotify.get(track_id) or "",
        "date": resolve_date(f"{title}{description}", timestamp).isoformat(),
        "words": words,
        "sentences": sentences,
        "sent_ratio": sent_ratio,
        # Per minute, so a 25-minute EBD and a 70-minute sermon are comparable.
        "words_min": words / (duration / 60),
        "sentences_min": sentences / (duration / 60),
        "score": clean.wav2vec_score(alignment),
    }


def main() -> None:
    config.ensure_dirs()
    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    todo = pending(sorted(config.RAW_DIR.glob("*.txt")), done_ids())
    print(f"raw transcripts to process: {len(todo)}", flush=True)
    if not todo:
        return

    print(f"LanguageTool: {'on' if clean.language_tool_available() else 'OFF (spaCy only)'}")
    preachers, full_names = preacher_matcher()
    spotify = spotify_ids.load()

    with config.ROWS_JSONL.open("a", encoding="utf-8") as out:
        for i, raw_path in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {raw_path.stem}", flush=True)
            try:
                row = build_row(raw_path, preachers, full_names, spotify)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                continue
            if row is None:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            below = " (below cutoff)" if row["score"] <= config.MIN_SCORE else ""
            print(f"  score {row['score']}{below}, {row['words']} words", flush=True)


if __name__ == "__main__":
    main()
