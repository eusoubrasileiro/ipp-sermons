# CLAUDE.md — tools/corpus-update

Supplements the root `CLAUDE.md`. Read `README.md` in this directory before
changing anything here — it is the authority on the pipeline and its failure
modes, and is not repeated here.

- **Interpreter is `${IPP_VENV_DIR:-/mnt/Data/venv/ipp-sermons}/bin/python`**,
  never system `python3`. This overrides the user-level rule for this directory:
  the venv carries spaCy-pt, LanguageTool and the NVIDIA libraries CTranslate2
  dlopens by soname.

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
