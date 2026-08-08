# Corpus update

Brings `data/` up to date with whatever the church has published since the last
run. This is the Python half; `pnpm corpus:update` drives it and then carries
on through the facet passes, indexing and the release.

```bash
pnpm corpus:update              # the whole thing, upstream to production
tools/corpus-update/run.sh      # just this half
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
| `transcribe` | ffmpeg pre-filter, then WhisperX `large-v3` in Portuguese on the GPU | skipping stems that already have a raw transcript, and stems handed to a peer |
| `postprocess` | spaCy/LanguageTool cleaning, metrics, score; writes the corpus `.txt` | skipping ids already in `rows.jsonl` |
| `append` | Appends new rows to `metadata.csv` | skipping ids already in the CSV |

Run one on its own with `run.sh <stage>`. `run.sh` stops at `append`: the order
of everything from `data/` inward lives in `scripts/corpus-update.sh`, and
stating it twice is how it came to be stated wrongly here.

## A second machine

Transcription is the long pole, and the only stage that parallelises across
boxes. `peer.sh` lends one:

```bash
tools/corpus-update/peer.sh dispatch predator      # hand it the back half, start it
tools/corpus-update/peer.sh status   predator
tools/corpus-update/peer.sh collect  predator      # transcripts home, assignment released
```

Audio and its `.info.json` go out, `raw/*.txt` and `alignment/*.gz` come back,
and nothing else moves. `discover`, `fetch`, `postprocess` and `append` stay on
the box that owns `data/`, for the reason the section above gives: those numbers
have to keep coming from one spaCy/LanguageTool install.

The sidecar is not cargo. `coverage()` divides the transcript's last timestamp
by the duration SoundCloud declared, and with no sidecar to ask it falls back to
measuring the `.wav` — which a truncated download passes against itself, every
time. Dispatch sent audio alone once, and a night of GPU came home as 27
well-formed transcripts of sermons that had only partly downloaded. `collect`
re-checks every transcript for the same reason, but by then the GPU time is
already spent.

What keeps the two boxes off each other's work is
`$IPP_WORK_DIR/assigned/<host>.txt`, written by `dispatch` and read by every
later `transcribe`. Splitting positionally instead — "you take the second half"
— quietly breaks: the backlog shrinks from the head as work completes, so the
second half slides backwards onto sermons the peer took hours ago, and both
boxes spend an hour of GPU on the same sermon. `collect` releases the
assignment, and only once the peer is actually idle.

A peer needs the repo, `ffmpeg`, any `python3`, and the ml-tools venv that
`whisperx_worker.py` runs under. It does **not** need the pipeline venv, yt-dlp,
spaCy or a copy of `data/`. Set `IPP_PEER_WORK_DIR` / `IPP_PEER_REPO` if its
layout differs from the defaults; a laptop has no `/mnt/Data`.

Being a laptop, it also has to not fall asleep mid-run: mask the sleep targets,
set `HandleLidSwitch=ignore`, and `loginctl enable-linger` so the run survives a
desktop logout.

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

**Compute type.** The corpus was transcribed at `float16`, and any GPU that
offers it keeps using it. CTranslate2 does not offer `float16` below compute
capability 7.0 — a GTX 1060 is 6.1 — and does not refuse either: it substitutes
`float32`, which for `large-v3` is 6.2 GB of weights on a 6 GB board, so the run
dies of an OOM hours in with nothing in the log connecting the two.
`whisperx_worker.py` therefore chooses explicitly — `float16` where the card has
it, `int8_float32` where it does not — and raises on a `--compute-type` the card
cannot honour. The run prints which type and which card produced its
transcripts, once, because with two machines that is the only record of it.

The fallback is corpus-comparable, measured rather than assumed. One 53-minute
sermon, GTX 1060 `int8_float32` against GTX 1660 SUPER `float16`: alignment
score 85.9 vs 85.8, identical coverage, 0.57% of words different — nearly all of
them spellings of one Hebrew proper noun (`Jeoás` / `Jehoás` / `Geoás`) that both
boxes already spell inconsistently *within a single transcript*. Speed was 24
min per audio-hour against 22: Pascal has int8 acceleration, so the older card
is not the handicap it looks like.

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
