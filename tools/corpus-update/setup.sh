#!/usr/bin/env bash
# One-time provisioning for the corpus update pipeline. Idempotent.
set -euo pipefail

VENV="${IPP_VENV_DIR:-/mnt/Data/venv/ipp-sermons}"

if [ ! -x "$VENV/bin/python" ]; then
  # 3.12 rather than the system 3.14: spaCy and its Portuguese model still lag
  # a release behind, and this box's other ML venvs are already on 3.12.
  uv venv --python 3.12 "$VENV"
fi

# pytest is here rather than in the workspace because this directory is outside
# it: `pnpm test` cannot see these files and must not be widened to.
uv pip install --python "$VENV/bin/python" \
  yt-dlp pandas rapidfuzz spacy language_tool_python requests click pytest

# VIRTUAL_ENV must be exported: `spacy download` shells out to `uv pip install`,
# which refuses to guess a target environment.
"$VENV/bin/python" -c "import spacy; spacy.load('pt_core_news_lg')" >/dev/null 2>&1 ||
  VIRTUAL_ENV="$VENV" "$VENV/bin/python" -m spacy download pt_core_news_lg

# language_tool_python drives a Java LanguageTool server and this machine has no
# system JVM. A private JRE keeps that out of the OS and avoids needing sudo;
# without it the cleaner still runs, just without the grammar passes.
if [ ! -x "$VENV/jre/bin/java" ]; then
  mkdir -p "$VENV/jre"
  curl -sSL -o "$VENV/jre/jre.tar.gz" \
    "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jre/hotspot/normal/eclipse"
  tar xzf "$VENV/jre/jre.tar.gz" -C "$VENV/jre" --strip-components=1
  rm "$VENV/jre/jre.tar.gz"
fi

echo "ready: $VENV"
"$VENV/jre/bin/java" -version
