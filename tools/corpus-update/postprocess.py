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

# `DD-MM-YYYY`, with any short run of non-digits as the separator so `/` and
# `-` both work. The two lookarounds are what keep it a date rather than three
# numbers that happen to be adjacent: without `(?!\d)` the corpus's `20223` and
# `20245` -- years typed with an extra digit -- read as 2022 and 2024, which is
# wrong by exactly one year and looks entirely right.
DATE_IN_TITLE_RE = re.compile(r"(?<!\d)(\d{1,2})\D{1,3}(\d{1,2})\D{1,3}(\d{4})(?!\d)")

# The same date, but only where the church labelled it as one. The description
# is a paragraph of free prose on most of the archive and a `Pastor:/Data:/
# Livro:/Assunto:` block on the newest uploads, so the label -- anchored to the
# start of its own line, `re.M` -- is what separates the field from the prose
# around it. Searching the description for digits instead is precisely the bug
# this fell into once already; see resolve_date.
DATE_LABEL_RE = re.compile(r"^[ \t]*Data:[ \t]*(\d{1,2})\D{1,3}(\d{1,2})\D{1,3}(\d{4})[ \t]*$", re.M)


def plausible_date(match: re.Match | None) -> datetime.date | None:
    """A `DD-MM-YYYY` match as a date, or None if it was never one.

    The window matches `resolveDate` in backend/src/lib/corpus.ts so the loader
    never has to second-guess a committed row.
    """
    if match is None:
        return None
    try:
        d, m, y = map(int, match.groups())
        parsed = datetime.date(y, m, d)
    except ValueError:
        return None
    return parsed if 2015 <= parsed.year <= 2030 else None


def resolve_date(title: str, description: str, timestamp: int) -> datetime.date:
    """The sermon's date: the title, then the description's label, then upload.

    Titles carry a `DD-MM-YYYY` prefix by convention, but only by convention:
    the corpus contains "05-01-20245", "0223-05-07" and a run of episodes with
    no date at all. The title stays first because 619 of the 622 committed rows
    were dated from it and this order is what keeps them still.

    The `Data:` label is second because the convention ended. The church
    stopped dating its titles and, in the same season, went from publishing 8-21
    days after the service to 77-91 -- so the upload timestamp, which used to be
    a fine backstop, is now most of a quarter wrong. The label is the church
    stating the date itself, and on those uploads it is the only source left
    that is right.

    Read through the label, never by searching. This used to search the title
    and the description concatenated with nothing between them, and a loose
    `(\\d{2}).+(\\d{2}).+(\\d{4})` over that manufactured dates out of digits
    that were never one -- "2 Reis 21_1-9" and whatever followed it. Three
    sermons went live under a month nobody preached them in. That fix (5027df9)
    then dropped the description wholesale and took the good field with the bad
    parse; the anchor is what lets it come back. A wrong date that passes every
    check is worse than no date, because the fallback is free.
    """
    for match in (DATE_IN_TITLE_RE.search(title), DATE_LABEL_RE.search(description)):
        parsed = plausible_date(match)
        if parsed is not None:
            return parsed

    return datetime.datetime.fromtimestamp(timestamp, datetime.UTC).date()


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

    A conference or a joint lesson credits several names in that tail
    ("por Pr. Bruno Melo, Seminarista Felipe Ricieri e André Cunha"), and the
    corpus has one `artist` column. Fuzzy-matching the whole run against a
    single name scores below the cutoff and the sermon ends up with no preacher
    at all, so each name is offered on its own after the full string, and the
    first one that matches wins.
    """
    candidates = []

    description = (info.get("description") or "").strip()
    if " por " in description:
        tail = description.rsplit(" por ", 1)[-1].splitlines()[0]
        candidates.append(tail)
        names = [n.strip() for n in re.split(r",| e ", tail)]
        candidates.extend(n for n in names if n and n != tail)

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


def audio_file_for(stem: str) -> str | None:
    """The `audio` column: the downloaded file's own name, as yt-dlp wrote it."""
    for p in config.AUDIO_DIR.glob(f"{glob_escape(stem)}.*"):
        if p.suffix.lower() in config.AUDIO_SUFFIXES:
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
    regex capture in `pending`, which is always a string, and `"111" not in
    {111}` is always true: without the conversion this stage is never resumable
    and every re-run re-cleans every transcript and rewrites every corpus .txt.
    """
    if not config.ROWS_JSONL.exists():
        return set()
    with config.ROWS_JSONL.open(encoding="utf-8") as f:
        return {str(json.loads(line)["id"]) for line in f if line.strip()}


def pending(raw_paths: list[pathlib.Path], have: set[str]) -> list[pathlib.Path]:
    """The raw transcripts still to clean, in the order given."""
    return [p for p in raw_paths if (m := config.STEM_ID_RE.match(p.stem)) and m["id"] not in have]


def display_title(stem: str, info: dict) -> str:
    """What the church called the sermon, not what the filesystem would take.

    yt-dlp sanitises its output template, so the stem has already lost any
    character a filesystem refuses: `:` arrives as `_` and `"` as the fullwidth
    `＂`. Reading the title back off the stem stored "Eclesiastes 6:7-12" as
    "Eclesiastes 6_7-12" -- wrong on the page, and worse in the facet pass,
    which parses the reference out of the title and cannot see a verse range
    through the underscore, so four sermons recorded a chapter and no verses.

    The sidecar carries the title as published, and `build_row` already reads it.
    Falling back to the stem when it does not: a sanitised name beats an empty
    one, and the stem is the only other record of what the sermon is called.
    """
    title = str(info.get("title") or "").strip()
    if title:
        return title
    match = config.STEM_ID_RE.match(stem)
    return match["title"] if match else stem


def transcript_filename(stem: str) -> str:
    """The `txt` column: a path, so it stays on the sanitised stem.

    Deliberately not `display_title`. The corpus already holds every transcript
    under its sanitised name, and a real title can contain `/`; this names a
    file on disk, while `name` is what a reader sees.
    """
    match = config.STEM_ID_RE.match(stem)
    return f"{match['title'] if match else stem}.txt"


def build_row(raw_path: pathlib.Path, preachers, full_names, spotify) -> dict | None:
    match = config.STEM_ID_RE.match(raw_path.stem)
    if not match:
        print(f"  skipping {raw_path.stem}: no soundcloud id in filename", flush=True)
        return None
    track_id = match["id"]

    info = config.info_json_for(raw_path.stem)
    if info is None:
        print(f"  skipping {raw_path.stem}: no .info.json", flush=True)
        return None

    title = display_title(raw_path.stem, info)

    alignment = config.ALIGNMENT_DIR / f"{raw_path.stem}.gz"
    if not alignment.exists():
        print(f"  skipping {raw_path.stem}: no alignment .gz, cannot score", flush=True)
        return None

    # The corpus keys transcripts by title alone; the id lives in the CSV.
    transcript_name = transcript_filename(raw_path.stem)
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
        "date": resolve_date(title, description, timestamp).isoformat(),
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
