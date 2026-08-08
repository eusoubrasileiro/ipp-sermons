#!/usr/bin/env bash
# Lend a second machine to the transcribe stage.
#
#   tools/corpus-update/peer.sh dispatch predator        # hand it the back half, start it
#   tools/corpus-update/peer.sh dispatch predator 3/3    # hand it the last third instead
#   tools/corpus-update/peer.sh status   predator
#   tools/corpus-update/peer.sh collect  predator        # transcripts home, assignment released
#
# Only the transcribe stage is shared, and only ever one way: audio and its
# .info.json go out, `raw/*.txt` and `alignment/*.gz` come back. Everything
# else -- discover, fetch, postprocess, append -- stays on the box that owns
# `data/`, because `words`, `sentences`, `sent_ratio` and `score` have to keep
# coming from one spaCy/LanguageTool install or new rows stop being comparable
# to the old ones.
#
# What makes this safe rather than a race is `$WORK/assigned/<host>.txt`: the
# audio handed to a peer is written down, and this box's transcribe stage skips
# anything listed there. Splitting positionally cannot work -- the backlog
# shrinks from the head, so "the second half" slides backwards onto sermons the
# peer took hours ago.
#
# The peer needs: the repo, ffmpeg, a python3, and the ml-tools venv that
# whisperx_worker.py runs under. It does NOT need the pipeline venv, yt-dlp,
# spaCy or a copy of data/. A card without float16 is fine -- the worker picks
# int8_float32 on its own, which is the corpus-comparable second choice.
set -euo pipefail

cd "$(dirname "$0")"
PY="${IPP_VENV_DIR:-/mnt/Data/venv/ipp-sermons}/bin/python"
WORK="${IPP_WORK_DIR:-/mnt/Data/ipp-sermons-work}"
ASSIGNED="$WORK/assigned"

# Where the peer keeps things. Relative to its own $HOME, because a laptop has
# no /mnt/Data and hard-coding this box's layout is what makes a second machine
# feel like a special case instead of a normal one.
PEER_WORK="${IPP_PEER_WORK_DIR:-ipp-sermons-work}"
PEER_REPO="${IPP_PEER_REPO:-Projects/side-projects/ipp-sermons}"

CMD="${1:-}"
HOST="${2:-}"
SHARE="${3:-2/2}"
[ -n "$CMD" ] && [ -n "$HOST" ] || { sed -n '2,6p' "$0"; exit 1; }

HANDOUT="$ASSIGNED/$HOST.txt"
say() { printf '\n=== %s ===\n' "$*"; }

# transcribe.py needs only the standard library on a peer (the yt-dlp import
# lives in fetch.py, which it no longer pulls in), so the system python is
# enough there and the pipeline venv is not required.
peer_python() {
  ssh "$HOST" "test -x \"\$HOME/${IPP_PEER_VENV:-nonexistent}/bin/python\" \
    && echo \"\$HOME/${IPP_PEER_VENV:-nonexistent}/bin/python\" || command -v python3"
}

peer_running() {
  ssh "$HOST" "pgrep -f 'transcribe\.py' >/dev/null" 2>/dev/null
}

case "$CMD" in

dispatch)
  [ -f "$HANDOUT" ] && {
    echo "$HOST already holds $(wc -l < "$HANDOUT") sermons -- collect before dispatching again" >&2
    exit 1
  }
  peer_running && { echo "$HOST is already transcribing" >&2; exit 1; }

  say "choosing $SHARE of the backlog"
  mkdir -p "$ASSIGNED"
  # The split is computed by the same tested function the stage itself uses,
  # over the same pending list, so what is handed out is exactly what this box
  # would otherwise have done.
  "$PY" - "$SHARE" > "$HANDOUT" <<'EOF'
import sys
import transcribe

index, count = (int(part) for part in sys.argv[1].split("/", 1))
for path in transcribe.shard(transcribe.pending_audio(), index, count):
    print(path.name)
EOF
  count=$(wc -l < "$HANDOUT")
  [ "$count" -gt 0 ] || { rm -f "$HANDOUT"; echo "nothing pending to hand out"; exit 0; }
  echo "$count sermons -> $HOST"

  say "sending audio"
  ssh "$HOST" "mkdir -p \"\$HOME/$PEER_WORK\"/{audio,wav,raw,alignment}"
  # --ignore-existing: a re-dispatch after a failure should not re-push
  # gigabytes the peer already has.
  rsync -a --ignore-existing --info=progress2 --files-from="$HANDOUT" \
    "$WORK/audio/" "$HOST:$PEER_WORK/audio/"

  # The .info.json sidecars go with it, and they are what let the peer judge
  # its own output: `coverage` divides by the duration SoundCloud declared, and
  # with no sidecar to ask it falls back to measuring the .wav -- which a
  # truncated download passes against itself, every time. Sending audio alone
  # once cost a night of GPU that came home as 27 well-formed transcripts of
  # sermons that had only partly downloaded.
  #
  # Listed one by one rather than globbed because a track whose sidecar is
  # missing is not a reason to abort a dispatch of fifty.
  sidecars=$(mktemp)
  trap 'rm -f "$sidecars"' EXIT
  while IFS= read -r name; do
    if [ -f "$WORK/audio/${name%.*}.info.json" ]; then
      printf '%s\n' "${name%.*}.info.json"
    fi
  done < "$HANDOUT" > "$sidecars"
  echo "$(wc -l < "$sidecars") sidecars"
  rsync -a --files-from="$sidecars" "$WORK/audio/" "$HOST:$PEER_WORK/audio/"

  say "starting $HOST"
  # setsid so it survives this ssh, the peer's desktop logout and a dropped
  # wifi link; the peer needs `loginctl enable-linger` for the last of those.
  ssh "$HOST" "cd \"\$HOME/$PEER_REPO/tools/corpus-update\" && \
    IPP_WORK_DIR=\"\$HOME/$PEER_WORK\" nohup setsid $(peer_python) transcribe.py \
      >> \"\$HOME/$PEER_WORK/transcribe.log\" 2>&1 < /dev/null & sleep 2"
  echo "dispatched. progress: tools/corpus-update/peer.sh status $HOST"
  ;;

status)
  if peer_running; then echo "$HOST: transcribing"; else echo "$HOST: idle"; fi
  [ -f "$HANDOUT" ] && echo "assigned: $(wc -l < "$HANDOUT") sermons"
  ssh "$HOST" "cd \"\$HOME/$PEER_WORK\" 2>/dev/null && \
    echo \"done there: \$(ls raw/*.txt 2>/dev/null | wc -l)\" && tail -3 transcribe.log"
  ;;

collect)
  say "fetching transcripts"
  # --ignore-existing in this direction too, and it matters more: a transcript
  # this box already produced is the authority, and must not be replaced by a
  # peer's copy of the same sermon should the two ever overlap.
  rsync -a --ignore-existing "$HOST:$PEER_WORK/raw/" "$WORK/raw/"
  rsync -a --ignore-existing "$HOST:$PEER_WORK/alignment/" "$WORK/alignment/"

  # The peer cannot judge its own output: dispatch ships audio without the
  # .info.json that says how long the sermon is, so `coverage` there falls back
  # to measuring the .wav and a truncated download validates against itself.
  # This is the one place every peer transcript passes through and the sidecars
  # live, so the check belongs here. It once let a night of work come home as
  # 27 well-formed transcripts of audio that had only partly downloaded.
  "$PY" -c 'import transcribe
for stem in transcribe.discard_incomplete():
    print(f"  incomplete, discarded: {stem}")'

  echo "raw: $(ls "$WORK"/raw/*.txt 2>/dev/null | wc -l) transcripts here now"

  if peer_running; then
    # Releasing the assignment while the peer is mid-sermon would let this box
    # start the very sermon it is working on. Collect is therefore safe to run
    # repeatedly during a run; it just does not free anything until the end.
    echo "$HOST is still transcribing -- assignment kept, run collect again when it is idle"
  else
    rm -f "$HANDOUT"
    echo "$HOST idle, assignment released"
  fi
  ;;

*)
  sed -n '2,6p' "$0"
  exit 1
  ;;
esac
