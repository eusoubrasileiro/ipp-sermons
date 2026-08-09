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
# Titles carry a DD-MM-YYYY prefix by convention only, so the publication
# timestamp backstops them. What matters is *when it refuses to believe a
# title*: a date invented from digits that were never a date is worse than
# falling back, because it is wrong and plausible at the same time and puts a
# sermon under the wrong month in /datas with nothing to show for it.

# 2026-07-12; a Sunday, and the date the church actually uploaded that sermon.
PUBLISHED = 1783868400


def test_a_dated_title_is_believed():
    assert postprocess.resolve_date("27-07-2025 - 2 Reis 10.1-17", PUBLISHED).isoformat() == (
        "2025-07-27"
    )


def test_a_title_with_no_date_falls_back_to_publication():
    """The three sermons this fixes were live under a month nobody preached
    them in: with the description concatenated on, `2 Reis 21_1-9` and the
    digits after it parsed as a date that passed every plausibility check."""
    title = "Culto - Rev. Bruno Melo - 2 Reis 21_1-9 - Fizeram Pior"
    assert postprocess.resolve_date(title, PUBLISHED).isoformat() == "2026-07-12"


def test_the_date_is_found_past_a_filename_prefix():
    """`20251228_001_28-12-2025 - EBD` — the upload tool's own prefix comes
    first, and reading digits greedily took the date out of that instead."""
    got = postprocess.resolve_date("20251228_001_28-12-2025 - EBD", PUBLISHED)
    assert got.isoformat() == "2025-12-28"


def test_five_digits_are_not_a_year():
    """The corpus contains `07-05-20223` and `05-01-20245`. Taking the first
    four digits of those yields a year that is wrong by one and looks right, so
    the publication date is the better answer."""
    assert postprocess.resolve_date("07-05-20223 - Efésios 3.7-8", PUBLISHED).isoformat() == (
        "2026-07-12"
    )


def test_an_impossible_date_in_a_title_falls_back():
    assert postprocess.resolve_date("31-02-2025 - Salmos 1", PUBLISHED).isoformat() == "2026-07-12"


def test_a_year_outside_the_archive_is_not_a_sermon_date():
    """Guards against a scripture reference or a quoted year being read as one."""
    assert postprocess.resolve_date("11-09-1973 - Salmos 1", PUBLISHED).isoformat() == "2026-07-12"


def test_slashes_work_as_well_as_dashes():
    assert postprocess.resolve_date("30/06/2019 - Aula 78", PUBLISHED).isoformat() == "2019-06-30"
