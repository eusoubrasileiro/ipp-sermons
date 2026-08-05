#!/usr/bin/env bash
# The whole corpus update, end to end. Safe to re-run: every stage skips work it
# has already done, so a second run in a row is a no-op and a run next month
# picks up only what the church published since.
#
#   tools/corpus-update/run.sh              # everything
#   tools/corpus-update/run.sh discover     # or any single stage
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

# The indexer reads OPENROUTER_API_KEY and DATABASE_URL from the environment and
# nothing in the repo loads .env for it, so a bare `run.sh` would get all the way
# through transcription and then fail at the last step.
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
    echo
    echo "=== verify ==="
    # The workspace's own loader gets the last word on what Python produced,
    # before the indexer starts spending money on embeddings.
    (cd ../.. && pnpm verify:corpus)
    echo
    echo "=== index ==="
    # Content-hash keyed, so re-running over the whole corpus only embeds the
    # chunks that are actually new.
    (cd ../.. && pnpm index)
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 1
    ;;
esac
