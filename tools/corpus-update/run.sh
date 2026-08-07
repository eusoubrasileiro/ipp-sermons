#!/usr/bin/env bash
# SoundCloud to data/. Safe to re-run: every stage skips work it has already
# done, so a second run in a row is a no-op and a run next month picks up only
# what the church published since.
#
#   tools/corpus-update/run.sh              # discover .. append
#   tools/corpus-update/run.sh discover     # or any single stage
#
# This is the Python half. `pnpm corpus:update` drives it and then carries on
# through the facet passes, indexing and the release.
#
# Transcription is the long pole -- roughly 20 minutes of GPU per hour of audio
# on this box -- so a large backlog is measured in days, not hours. Interrupting
# it costs at most the sermon in flight.
set -euo pipefail

cd "$(dirname "$0")"
PY="${IPP_VENV_DIR:-/mnt/Data/venv/ipp-sermons}/bin/python"
STAGE="${1:-all}"
WORK="${IPP_WORK_DIR:-/mnt/Data/ipp-sermons-work}"

# One run at a time. Transcription takes days, so it is easy to forget one is
# already going; a second loop does not queue behind the first, it competes with
# it for the 6 GB of VRAM and both OOM.
mkdir -p "$WORK"
exec 9>"$WORK/.lock"
flock -n 9 || { echo "another corpus-update run holds $WORK/.lock" >&2; exit 1; }

# postprocess reads the Spotify credentials from the environment to fill
# sp_suffix_url, and nothing in this directory loads .env for it.
if [ -f ../../.env ]; then
  set -a
  # shellcheck disable=SC1091
  . ../../.env
  set +a
fi

run() { echo; echo "=== $1 ==="; "$PY" "$1.py"; }

case "$STAGE" in
  discover|fetch|transcribe|postprocess|append) run "$STAGE" ;;
  all)
    run discover
    run fetch
    run transcribe
    run postprocess
    run append
    # Stops at the seam. Everything from data/ inward -- verify, the facet
    # passes, indexing, the release -- belongs to scripts/corpus-update.sh,
    # which is where the order between those steps is written down. This used
    # to run verify:corpus and index here, which was a second statement of that
    # order and a wrong one: it skipped the facet half entirely and indexed
    # before the facets existed.
    echo
    echo "data/ is up to date. Next: pnpm corpus:update facets"
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 1
    ;;
esac
