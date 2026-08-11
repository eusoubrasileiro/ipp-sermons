"""The resume key of the postprocess stage.

This directory is outside the pnpm workspace, so `pnpm test` never sees it; the
gate here is pytest, run from the pipeline's own venv. It stays cheap on purpose
-- no spaCy, no GPU, no network -- because `scripts/corpus-update.sh` runs it as
a pre-flight before a transcription run that takes days.
"""

import json
import pathlib

import config
import postprocess


def write_rows(tmp_path: pathlib.Path, monkeypatch, *rows: dict) -> pathlib.Path:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    monkeypatch.setattr(config, "ROWS_JSONL", path)
    return path


def test_done_ids_reads_back_the_int_that_build_row_wrote(tmp_path, monkeypatch):
    """`build_row` writes `"id": int(track_id)`, and it has to keep doing that --
    `append.py`'s DTYPES are what stop the appended CSV line from formatting
    differently to the 455 rows already in metadata.csv. So the reading side is
    the side that has to convert."""
    write_rows(tmp_path, monkeypatch, {"id": 2039084780, "name": "CFW 21"})

    assert postprocess.done_ids() == {"2039084780"}


def test_done_ids_is_empty_when_nothing_has_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROWS_JSONL", tmp_path / "absent.jsonl")

    assert postprocess.done_ids() == set()


def test_pending_skips_a_stem_already_in_rows_jsonl(tmp_path):
    """The failure this guards: the ids never matched, so every re-run re-cleaned
    every raw transcript and rewrote every data/transcripts/*.txt -- a whole
    corpus of spurious diffs, on a critical path, from a stage documented as a
    no-op the second time."""
    done = tmp_path / "12-01-2025 - Romanos 8 [111].txt"
    fresh = tmp_path / "19-01-2025 - Romanos 9 [222].txt"

    assert postprocess.pending([done, fresh], {"111"}) == [fresh]


def test_pending_drops_a_stem_with_no_soundcloud_id(tmp_path):
    assert postprocess.pending([tmp_path / "sem id.txt"], set()) == []


PREACHERS = ["bruno melo", "eder mota"]
FULL_NAMES = ["Reverendo Bruno Melo", "Presbítero Éder Mota"]


def test_resolve_artist_reads_the_house_style():
    info = {"description": "1 Reis 19 - O plano de Deus por Rev. Bruno Melo"}

    assert postprocess.resolve_artist(info, PREACHERS, FULL_NAMES) == "Reverendo Bruno Melo"


def test_resolve_artist_takes_the_first_of_several_preachers():
    """Conferences and joint lessons credit more than one name, and the corpus
    has a single `artist` column. Fuzzy-matching the whole run of names against
    one name scores below the cutoff and the sermon ends up with no preacher at
    all -- which is what happened to the Maranhão lesson."""
    info = {
        "description": "Viagem Maranhão por Pr. Bruno Melo, "
        "Seminarista Felipe Ricieri e André Cunha"
    }

    assert postprocess.resolve_artist(info, PREACHERS, FULL_NAMES) == "Reverendo Bruno Melo"


def test_resolve_artist_leaves_it_blank_rather_than_guessing():
    info = {"description": "Uma aula por Alguém Que Não Está Na Lista"}

    assert postprocess.resolve_artist(info, PREACHERS, FULL_NAMES) is None


# --- the sermon's date ------------------------------------------------------
#
# Three sources, and the whole design is in the order: the title's DD-MM-YYYY
# prefix, then a `Data:` line in the SoundCloud description, then the
# publication timestamp. What matters is *when it refuses to believe a source*:
# a date invented from digits that were never a date is worse than falling
# back, because it is wrong and plausible at the same time and puts a sermon
# under the wrong month in /datas with nothing to show for it.

# 2026-07-12; a Sunday, and the date the church actually uploaded that sermon.
PUBLISHED = 1783868400

# The house style the church switched to once it stopped dating its titles.
HOUSE_STYLE = (
    "Pastor: Rev. Bruno Melo\nData: 26-04-2026\nLivro: 2 Reis\n"
    "Assunto: 2 Reis 21:1-9 - Fizeram Pior"
)


def test_a_dated_title_is_believed():
    assert postprocess.resolve_date("27-07-2025 - 2 Reis 10.1-17", "", PUBLISHED).isoformat() == (
        "2025-07-27"
    )


def test_the_date_is_found_past_a_filename_prefix():
    """`20251228_001_28-12-2025 - EBD` — the upload tool's own prefix comes
    first, and reading digits greedily took the date out of that instead."""
    got = postprocess.resolve_date("20251228_001_28-12-2025 - EBD", "", PUBLISHED)
    assert got.isoformat() == "2025-12-28"


def test_five_digits_are_not_a_year():
    """The corpus contains `07-05-20223` and `05-01-20245`. Taking the first
    four digits of those yields a year that is wrong by one and looks right, so
    the publication date is the better answer."""
    assert postprocess.resolve_date("07-05-20223 - Efésios 3.7-8", "", PUBLISHED).isoformat() == (
        "2026-07-12"
    )


def test_an_impossible_date_in_a_title_falls_back():
    got = postprocess.resolve_date("31-02-2025 - Salmos 1", "", PUBLISHED)
    assert got.isoformat() == "2026-07-12"


def test_a_year_outside_the_archive_is_not_a_sermon_date():
    """Guards against a scripture reference or a quoted year being read as one."""
    got = postprocess.resolve_date("11-09-1973 - Salmos 1", "", PUBLISHED)
    assert got.isoformat() == "2026-07-12"


def test_slashes_work_as_well_as_dashes():
    got = postprocess.resolve_date("30/06/2019 - Aula 78", "", PUBLISHED)
    assert got.isoformat() == "2019-06-30"


def test_the_description_carries_the_date_once_the_title_stopped():
    """The church dropped the DD-MM-YYYY title prefix and started publishing
    ~90 days late in the same season, so the publication timestamp is no longer
    a usable backstop -- it is a quarter of a year wrong. `Data:` is the church
    saying the date itself, and it is the only remaining source that is right."""
    title = "Culto - Rev. Bruno Melo - 2 Reis 21_1-9 - Fizeram Pior"

    got = postprocess.resolve_date(title, HOUSE_STYLE, PUBLISHED)

    assert got.isoformat() == "2026-04-26"


def test_the_title_outranks_the_description():
    """Both are the church's own words, but the title is the convention 619 of
    the 622 rows were built on. Nothing about this change may re-date them."""
    got = postprocess.resolve_date("27-07-2025 - 2 Reis 10.1-17", HOUSE_STYLE, PUBLISHED)

    assert got.isoformat() == "2025-07-27"


def test_a_scripture_reference_in_the_description_is_never_a_date():
    """The regression that `5027df9` fixed, and the reason the description is
    read through a label rather than searched. `2 Reis 21_1-9` followed by
    anything four-digit used to parse as a date that passed every plausibility
    check, and three sermons went live under a month nobody preached them in."""
    description = "2 Reis 21_1-9 - Fizeram Pior\nCulto vespertino, 19 30 2026 pessoas"

    got = postprocess.resolve_date("Culto - Rev. Bruno Melo", description, PUBLISHED)

    assert got.isoformat() == "2026-07-12"


def test_the_label_has_to_own_its_line():
    """Anchoring to the start of a line is what makes the label a label. Loose,
    `Data` is a common enough Portuguese word to appear mid-sentence next to
    digits that are not this sermon's date."""
    description = "Assunto: a data do próximo congresso é 10-10-2026\nLivro: 2 Reis"

    got = postprocess.resolve_date("Culto - Rev. Bruno Melo", description, PUBLISHED)

    assert got.isoformat() == "2026-07-12"


def test_an_implausible_labelled_date_falls_back_like_any_other():
    description = "Pastor: Rev. Lucas Antunes\nData: 31-02-2026\nLivro: Eclesiastes"

    got = postprocess.resolve_date("Culto", description, PUBLISHED)

    assert got.isoformat() == "2026-07-12"


def test_neither_source_leaves_the_publication_date():
    assert postprocess.resolve_date("Culto", "Pastor: Rev. X", PUBLISHED).isoformat() == (
        "2026-07-12"
    )


# Rows whose committed `date` is not what today's rule produces. Both predate
# this change -- the rule already disagreed with them and still does, by the
# same amount -- so they are pinned rather than fixed: correcting a date is a
# `data/**` diff, a re-index and a release, which is not what a date rule is
# allowed to drag behind it.
#
#   1519141552  `07-05-20223`, the five-digit year. Committed as `0223-05-07`
#               by the pipeline that predates the lookarounds. Harmless in
#               production: year 223 fails `resolveDate`'s window in
#               backend/src/lib/corpus.ts and the loader falls back to the
#               upload date, which is right.
#   2240497778  `20251228_001_28-12-2025`, committed as `2025-12-20` -- the
#               upload tool's own filename prefix was read as the date before
#               `test_the_date_is_found_past_a_filename_prefix`. This one is
#               live and wrong: `2025-12-20` is a plausible date, so the loader
#               has no reason to doubt it, and the page contradicts its title
#               by eight days.
STALE_COMMITTED_DATES = {"1519141552", "2240497778"}


def test_the_rule_and_the_committed_corpus_still_agree():
    """The assertion that authorises the change: re-deriving every date in
    `data/metadata.csv` reproduces the committed column, except on rows already
    known to have drifted.

    Reading a source of truth is what a unit test is normally not allowed to do,
    and the exception is deliberate. A date rule is only safe if it agrees with
    every date already published, and no fixture can stand in for 622 of them --
    the risk being guarded is a rule that looks right on invented input and
    silently re-dates the archive. 263 KB and no network, so it stays inside the
    pre-flight budget this file is written to.

    Pinned as an exact set, so it fails in both directions: a new disagreement
    is a regression, and repairing one of these rows has to come here and say so.

    Rows with an empty `date` are the six from 2019 that predate the column;
    nothing has re-processed them and this change is not what will.
    """
    import csv

    moved = {}
    with config.METADATA_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["date"] or not row["timestamp"]:
                continue
            got = postprocess.resolve_date(
                row["name"], row["description"] or "", int(row["timestamp"])
            )
            if got.isoformat() != row["date"]:
                moved[row["id"]] = (row["name"], row["date"], got.isoformat())

    assert set(moved) == STALE_COMMITTED_DATES, moved


def test_reading_the_label_only_ever_recovers_a_committed_date():
    """The other half, and the one that says what the change is *for*.

    Under the title-only rule these three rows re-date themselves on any
    re-processing -- to 2026-07-12, 2026-08-02 and 2026-07-20, the days they
    were uploaded, 77 to 91 days after they were preached. The label puts each
    of them back on exactly the date already committed. So the change cannot
    move the archive; it can only stop the archive from moving.
    """
    import csv

    recovered = []
    with config.METADATA_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            description = row["description"] or ""
            if not postprocess.DATE_LABEL_RE.search(description):
                continue
            got = postprocess.resolve_date(row["name"], description, int(row["timestamp"]))
            title_only = postprocess.resolve_date(row["name"], "", int(row["timestamp"]))
            recovered.append((got.isoformat(), title_only.isoformat(), row["date"]))

    assert recovered, "no row carries a `Data:` label; this rule guards nothing"
    assert all(got == committed for got, _, committed in recovered), recovered
    assert all(title_only != committed for _, title_only, committed in recovered), recovered
