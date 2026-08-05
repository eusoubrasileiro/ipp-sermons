"""Step 5 -- append the new rows to data/metadata.csv.

Strictly additive: existing lines are never re-read and re-written. Round-tripping
the whole file through pandas changes the last digit of a few float columns
(parse and format are not exact inverses), which would show up as a diff on
sermons nobody touched and bury the real change. Appending sidesteps it.

Rows already present by SoundCloud id are skipped, so this is safe to re-run.
"""

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


def main() -> None:
    if not config.ROWS_JSONL.exists():
        print("no rows.jsonl; run postprocess.py first")
        return

    with config.ROWS_JSONL.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    existing = pd.read_csv(config.METADATA_CSV)
    known = set(existing["id"].astype(str))
    new = [r for r in rows if str(r["id"]) not in known]

    print(f"rows.jsonl {len(rows)}, already in metadata.csv {len(rows) - len(new)}, appending {len(new)}")
    if not new:
        return

    df = pd.DataFrame(new)[list(existing.columns)]
    for col, dtype in DTYPES.items():
        df[col] = df[col].astype(dtype)

    text = config.METADATA_CSV.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    text += df.to_csv(index=False, header=False)
    config.METADATA_CSV.write_text(text, encoding="utf-8")

    kept = sum(1 for r in new if r["score"] > config.MIN_SCORE)
    print(f"appended {len(new)} rows; {kept} clear the score>{config.MIN_SCORE} cutoff")


if __name__ == "__main__":
    main()
