# CLAUDE.md — tools/corpus-update

Supplements the root `CLAUDE.md`. The pipeline's reasoning lives in the module
and function docstrings, at the point of each decision — read those. Only rules
that no single file owns are repeated here.

- **A short transcript is evidence about the audio first.** `coverage()` fails
  in `transcribe.py`, but the usual cause is in `fetch.py`: a download that lost
  HLS fragments is transcribed completely and still reports 86%, because the
  denominator is the sermon's declared length. That looks like a deterministic
  model failure and is not one. Run `content_seconds()` against the sidecar
  before touching the VAD, the batch size or `MIN_COVERAGE` — half a second,
  against twenty minutes of GPU for a re-run that fails identically.

- **Interpreter is `${IPP_VENV_DIR:-/mnt/Data/venv/ipp-sermons}/bin/python`**,
  never system `python3`. This overrides the user-level rule for this directory:
  the venv carries spaCy-pt, LanguageTool and the NVIDIA libraries CTranslate2
  dlopens by soname.

  The one exception is a **peer** running `transcribe.py` alone (`peer.sh`):
  that stage needs only the standard library, because the real work happens in
  the ml-tools venv through `whisperx_worker.py`. Keep it that way — importing
  `fetch` there would drag yt-dlp onto a machine that never downloads anything,
  which is why `AUDIO_SUFFIXES` lives in `config.py`.

- **This directory is outside the pnpm workspace** (`pnpm-workspace.yaml`) and
  outside Biome's `includes`. `pnpm lint`, `pnpm test` and `pnpm quality-gate`
  do not see it, and must not be widened to. The gates on this code are its own
  `pytest` (run from the venv) and `pnpm verify:corpus`, which loads the output
  through the real `loadSermons()`.

- **Never recompute `words`, `sentences`, `sent_ratio` or `score` outside this
  venv.** Those numbers are whatever spaCy-pt, LanguageTool and WhisperX's
  wav2vec alignment produce. Different tooling silently makes new rows
  incomparable to the existing corpus — the failure this pipeline exists to
  prevent.
