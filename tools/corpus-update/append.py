"""Step 5 -- put the new and corrected rows into data/metadata.csv.

Line-oriented, not table-oriented: a row is appended or its own line is
rewritten, and every other byte of the file is left exactly as it was.
Round-tripping the whole thing through pandas changes the last digit of a few
float columns (parse and format are not exact inverses), which shows up as a
diff on sermons nobody touched and buries the real change.

A known id used to be skipped outright, which made a re-run safe and a repair
impossible. It has to be possible: a download that lost HLS fragments yields a
complete-looking row built from partial audio, and the cure is to fetch the
sermon again and put the corrected numbers where the wrong ones are. So a known
id is now compared, and rewritten only when it actually differs -- which keeps
the re-run a no-op, since postprocess's rows.jsonl accumulates every sermon it
has ever cleaned.
"""

import csv
import io
import json

import pandas as pd

import config

# Types pandas infers when it reads the existing file. Setting them explicitly
# is what makes the appended line format identically -- ints without a ".0",
# booleans as True/False, and empty strings for missing text.
DTYPES = {
    "wav": bool,
    "transcribed": bool,
    "processed": bool,
    "id": "int64",
    "view_count": "int64",
    "timestamp": "int64",
    "words": "int64",
    "sentences": "int64",
    "duration": "float64",
    "sent_ratio": "float64",
    "words_min": "float64",
    "sentences_min": "float64",
    "score": "float64",
}


def as_csv_lines(rows: list[dict], columns: list[str]) -> list[str]:
    """Format rows the way the file already formats them.

    Through pandas rather than `csv` directly, because DTYPES is what makes an
    id print as `2139294189` and not `2139294289.0`, and what quotes a
    description containing a comma the same way the existing 621 lines are
    quoted.
    """
    df = pd.DataFrame(rows)[columns]
    for col, dtype in DTYPES.items():
        if col in df:
            df[col] = df[col].astype(dtype)
    return records(df.to_csv(index=False, header=False, lineterminator="\n"))


def records(text: str) -> list[str]:
    """CSV records, which are not lines.

    `name` and `description` carry the line breaks the church typed into them,
    so 622 sermons occupy 680 lines of metadata.csv. Splitting on newlines finds
    an id on some of those lines and not others, and every sermon whose
    description wraps then looks absent -- so it is appended a second time.
    """
    out, start, quoted = [], 0, False
    for i, char in enumerate(text):
        if char == '"':
            # A doubled quote inside a quoted field toggles twice and lands back
            # where it started, which is the escape rule, so this needs no case.
            quoted = not quoted
        elif char == "\n" and not quoted:
            out.append(text[start:i])
            start = i + 1
    if text[start:]:
        out.append(text[start:])
    return out


def fields(line: str) -> list[str]:
    """A CSV line's values.

    Parsed rather than split on commas: `name` and `description` are free text
    and routinely contain them. Comparing these rather than the raw lines is
    what keeps a re-run from "correcting" a row whose only difference is that
    the writer chose to quote a field and we did not -- which would rewrite
    most of the corpus the first time this ran.
    """
    return next(csv.reader(io.StringIO(line)), [])


def main() -> None:
    if not config.ROWS_JSONL.exists():
        print("no rows.jsonl; run postprocess.py first")
        return

    with config.ROWS_JSONL.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    lines = records(config.METADATA_CSV.read_text(encoding="utf-8"))
    columns = fields(lines[0])
    id_column = columns.index("id")
    at = {parsed[id_column]: i for i, line in enumerate(lines[1:], 1)
          if len(parsed := fields(line)) > id_column}

    formatted = dict(zip((str(r["id"]) for r in rows), as_csv_lines(rows, columns)))
    new = [sid for sid in formatted if sid not in at]
    fixed = [sid for sid in formatted
             if sid in at and fields(formatted[sid]) != fields(lines[at[sid]])]

    print(
        f"rows.jsonl {len(rows)}, already in metadata.csv {len(rows) - len(new)}, "
        f"appending {len(new)}, correcting {len(fixed)}"
    )
    if not new and not fixed:
        return

    for sid in fixed:
        lines[at[sid]] = formatted[sid]
    lines.extend(formatted[sid] for sid in new)
    config.METADATA_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    kept = sum(1 for r in rows if str(r["id"]) in new and r["score"] > config.MIN_SCORE)
    print(f"appended {len(new)} rows; {kept} clear the score>{config.MIN_SCORE} cutoff")
    if fixed:
        print(f"corrected {len(fixed)} rows already in the corpus")


if __name__ == "__main__":
    main()
