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
