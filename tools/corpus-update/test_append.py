"""Appending to, and correcting, data/metadata.csv.

Same contract as the rest: no network, no ffprobe, no model. The CSV is written
and read back as text, because the property under test is textual -- which bytes
of the file changed.

The stage was strictly additive, which was right while the corpus could only
grow. It cannot only grow: a download that lost HLS fragments produces a whole
row from partial audio, and the only cure is to fetch it again and put the
corrected numbers where the wrong ones are. Skipping a known id, which is what
made a re-run safe, is exactly what made a repair impossible.

What must survive the change is the reason the stage appends in the first place:
round-tripping the whole file through pandas moves the last digit of a few float
columns, which shows up as a diff on sermons nobody touched. So a repair rewrites
its own line and leaves every other byte alone.
"""

import json

import config

import append

COLUMNS = ["name", "id", "words", "score", "description"]


def row(name: str, sermon_id: int, words: int, score: float, description: str = "x") -> dict:
    return {"name": name, "id": sermon_id, "words": words, "score": score,
            "description": description}


def corpus(tmp_path, monkeypatch, *rows: dict) -> None:
    """A metadata.csv with these rows, and an empty rows.jsonl beside it."""
    csv = tmp_path / "metadata.csv"
    records = [",".join(COLUMNS)]
    for r in rows:
        records.append(
            ",".join(f'"{r[c]}"' if c == "description" else str(r[c]) for c in COLUMNS)
        )
    csv.write_text("\n".join(records) + "\n", encoding="utf-8")
    monkeypatch.setattr(config, "METADATA_CSV", csv)
    monkeypatch.setattr(config, "ROWS_JSONL", tmp_path / "rows.jsonl")
    monkeypatch.setattr(append, "DTYPES", {"id": "int64", "words": "int64", "score": "float64"})


def produced(*rows: dict) -> None:
    config.ROWS_JSONL.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def lines() -> list[str]:
    return config.METADATA_CSV.read_text(encoding="utf-8").splitlines()


def test_a_sermon_the_corpus_does_not_have_is_appended(tmp_path, monkeypatch):
    corpus(tmp_path, monkeypatch, row("a", 1, 100, 80.0))
    produced(row("a", 1, 100, 80.0), row("b", 2, 200, 90.0))

    append.main()

    assert [line.split(",")[0] for line in lines()[1:]] == ["a", "b"]


def test_a_re_run_that_produced_nothing_new_touches_nothing(tmp_path, monkeypatch):
    corpus(tmp_path, monkeypatch, row("a", 1, 100, 80.0))
    before = config.METADATA_CSV.read_text(encoding="utf-8")
    produced(row("a", 1, 100, 80.0))

    append.main()

    assert config.METADATA_CSV.read_text(encoding="utf-8") == before


def test_a_re_transcribed_sermon_replaces_its_own_row(tmp_path, monkeypatch):
    """The repair. The first row was built from audio missing seven minutes, so
    its word count and score describe a sermon nobody preached."""
    corpus(tmp_path, monkeypatch, row("a", 1, 100, 80.0), row("b", 2, 5000, 86.5))
    produced(row("b", 2, 6971, 86.5))

    append.main()

    assert len(lines()) == 3
    assert lines()[2].startswith("b,2,6971,")


def test_a_replacement_leaves_every_other_line_byte_identical(tmp_path, monkeypatch):
    """Why this stage never round-tripped the file through pandas: parse and
    format are not exact inverses, so rewriting untouched rows moves the last
    digit of a float and buries the real change in noise."""
    unrelated = row("a", 1, 100, 1.045016077170418)
    corpus(tmp_path, monkeypatch, unrelated, row("b", 2, 5000, 86.5))
    before = lines()
    produced(row("b", 2, 6971, 86.5))

    append.main()

    assert lines()[0] == before[0]
    assert lines()[1] == before[1]
    assert lines()[2] != before[2]


def test_a_replacement_formats_like_the_line_it_replaces(tmp_path, monkeypatch):
    """Ints without a trailing `.0`, and the column order of the file rather
    than of the dict -- the same reason DTYPES exists for appended rows."""
    corpus(tmp_path, monkeypatch, row("b", 2, 5000, 86.5))
    produced({"score": 91.25, "words": 6971, "id": 2, "name": "b", "description": "x"})

    append.main()

    assert lines()[1].split(",")[:4] == ["b", "2", "6971", "91.25"]


def test_a_description_with_a_newline_in_it_is_still_one_row(tmp_path, monkeypatch):
    """A CSV record is not a line. SoundCloud descriptions carry the line breaks
    the church typed into them, so 622 sermons occupy 680 lines of the file.

    Reading it a line at a time finds an id on some of them and not others, and
    every sermon whose description wraps looks absent -- so `append` adds it a
    second time. That is 50 duplicate sermons out of 622, from one `splitlines`.
    """
    wrapped = row("a", 1, 100, 80.0, description="por Pastor Bruno\nsegunda linha")
    corpus(tmp_path, monkeypatch, wrapped, row("b", 2, 5000, 86.5))
    before = config.METADATA_CSV.read_text(encoding="utf-8")
    produced(wrapped, row("b", 2, 5000, 86.5))

    append.main()

    assert config.METADATA_CSV.read_text(encoding="utf-8") == before


def test_a_repair_beside_a_wrapped_description_still_lands(tmp_path, monkeypatch):
    corpus(tmp_path, monkeypatch,
           row("a", 1, 100, 80.0, description="linha um\nlinha dois"),
           row("b", 2, 5000, 86.5))
    produced(row("b", 2, 6971, 86.5))

    append.main()

    import pandas as pd
    df = pd.read_csv(config.METADATA_CSV)
    assert len(df) == 2
    assert df.loc[df.id == 2, "words"].item() == 6971
    assert df.loc[df.id == 1, "description"].item() == "linha um\nlinha dois"


def test_a_row_that_did_not_change_is_not_rewritten(tmp_path, monkeypatch):
    """postprocess is resumable, so rows.jsonl accumulates every sermon it ever
    cleaned. Treating all of them as repairs would rewrite the whole corpus on
    every run -- which is the diff noise this stage exists to avoid."""
    corpus(tmp_path, monkeypatch, row("a", 1, 100, 80.0), row("b", 2, 5000, 86.5))
    before = config.METADATA_CSV.read_text(encoding="utf-8")
    produced(row("a", 1, 100, 80.0), row("b", 2, 5000, 86.5))

    append.main()

    assert config.METADATA_CSV.read_text(encoding="utf-8") == before
