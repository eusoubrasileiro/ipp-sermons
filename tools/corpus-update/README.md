# Corpus update

Brings `data/` up to date with whatever the church has published since the last
run, and hands off to the existing indexer.

```bash
tools/corpus-update/run.sh
```

Idempotent and incremental at every stage. Running it twice does nothing the
second time; running it next month picks up only what is new.

## Why it is split in two languages

Python owns everything from SoundCloud down to `data/metadata.csv`, and
TypeScript owns everything from `data/` inward.

That is not a preference, it is where the output format is decided. `words`,
`sentences`, `sent_ratio` and `score` are not conventions we chose — they are
whatever spaCy's Portuguese model, LanguageTool's rules and WhisperX's wav2vec
alignment happen to produce, and all three are Python-only. Recomputing them in
TypeScript would give different numbers for new sermons than for the 455 already
in the corpus, which is exactly the failure this pipeline exists to avoid.

The seam is the two flat files the workspace already treats as its input —
`data/metadata.csv` and `data/transcripts/*.txt` — and the workspace still gets
the last word on them: `pnpm verify:corpus` loads the result through the real
`loadSermons()` and fails the run before the indexer spends anything.

## Stages

| Stage | What it does | Resumes by |
|---|---|---|
| `discover` | Flat-lists the SoundCloud channel, diffs track ids against `metadata.csv` | recomputed each run |
| `fetch` | Downloads audio + `.info.json` per track | skipping ids that already have an audio file |
| `transcribe` | ffmpeg pre-filter, then WhisperX `large-v3` in Portuguese on the GPU | skipping stems that already have a raw transcript |
| `postprocess` | spaCy/LanguageTool cleaning, metrics, score; writes the corpus `.txt` | skipping ids already in `rows.jsonl` |
| `append` | Appends new rows to `metadata.csv` | skipping ids already in the CSV |
| verify + index | `pnpm verify:corpus`, then `pnpm index` | the indexer is content-hash keyed |

Run one on its own with `run.sh <stage>`.

## Where things live

Audio never enters the repo and never goes to the VPS. Everything derived from
it stays in the work directory, `$IPP_WORK_DIR` (default
`/mnt/Data/ipp-sermons-work`):

```
audio/       downloaded media + yt-dlp .info.json   (~170 MB per sermon)
wav/         16 kHz mono, deleted after transcription
raw/         WhisperX output, before cleaning
alignment/   word-level confidence .gz, the source of `score`
pending.json what discover found missing
rows.jsonl   finished rows, before they are appended to the CSV
```

Only the cleaned `.txt` and the new CSV rows cross back into git.

## Setup

`setup.sh` is one-time and idempotent. It builds the venv, installs the spaCy
Portuguese model, and unpacks a private Temurin JRE, because
`language_tool_python` drives a Java server and this box has no system JVM. If
that JRE is missing the cleaner says so and runs the spaCy passes only — the
transcripts are still segmented and capitalised, just not grammar-corrected.

Transcription reuses the WhisperX install at
`~/Projects/amiticia/repositories/tools/ml-tools/audio`, which is already pinned
to `large-v3` / float16 / pt against this GPU and emits the word-level alignment
the score depends on. It needs the venv's bundled NVIDIA libraries on
`LD_LIBRARY_PATH`; `transcribe.py` sets that up, since CTranslate2 dlopens them
by soname and will not find them otherwise.

## Things worth knowing

**Dates.** Titles carry a `DD-MM-YYYY` prefix by convention only. The corpus
contains `05-01-20245`, `0223-05-07`, and episodes with no date at all. The
original pipeline wrote whatever its regex produced and left the column empty
otherwise, which is where the six unusable dates came from. `resolve_date` falls
back to the SoundCloud publication timestamp whenever the title is missing or
implausible, and the timestamp is always written.

**Score.** `score` is mean wav2vec alignment confidence, not a quality judgement
about the preaching. The indexer keeps `score > 50`; below that the audio is bad
enough that the transcript is not worth searching. New rows are written
regardless and the loader filters them, same as the existing corpus.

**Spotify.** `sp_suffix_url` is best-effort and needs no credential: it comes
from a cached scrape first, and from the API only if `SPOTIFY_CLIENT_ID` /
`SPOTIFY_CLIENT_SECRET` are set. The secret committed in the archive notebooks
is public and must be rotated; do not reuse it.

**SoundCloud throttling.** Fetch runs 2 tracks at a time with 4 concurrent
fragments. Higher and SoundCloud starts answering 403 to the fragment manifests
part-way through the backlog.

**Truncated transcripts.** WhisperX sometimes returns having transcribed only
the first few minutes of a sermon, because the silero VAD decides the rest is
not speech. Alignment then succeeds on those few segments, so the result is a
well-formed transcript of a fifth of the sermon that scores fine and would index
fine — nothing downstream can tell. `transcribe.py` therefore checks how far the
last aligned segment reaches into the audio and rejects anything under 90%.

The original pipeline used the pyannote VAD, which does not have this problem,
and this was tried first. It does not fit: on a 6 GB card, pyannote's
segmentation model plus large-v3 in float16 OOMs before transcribing a word.
(It also needs `torch.load` forced back to `weights_only=False` under torch
2.6 — see `whisperx_worker.py`.) So silero is what runs, and the misfire is
handled by detection and retry rather than avoided. It is intermittent, not
deterministic, so a retry generally clears it.
