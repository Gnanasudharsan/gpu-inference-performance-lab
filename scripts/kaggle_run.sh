#!/usr/bin/env bash
# Run the NVIDIA T4 measurement session on Kaggle (free tier) and ingest the results.
#
# Prerequisites (one-time, done by the account owner):
#   1. Kaggle account with phone verification (needed for internet + GPU in kernels).
#   2. API credentials from kaggle.com -> Settings -> API, either:
#      - new API token  -> save to ~/.kaggle/access_token (chmod 600) and export KAGGLE_USERNAME
#      - legacy API key -> save the downloaded kaggle.json to ~/.kaggle/kaggle.json (chmod 600)
#   3. The commit to measure is pushed to GitHub (the kernel clones it by SHA).
#
# Usage: scripts/kaggle_run.sh            # push, run (<= 3 h), wait, download, analyze
#        scripts/kaggle_run.sh --fetch    # only download + ingest the latest run
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
k() { uvx --quiet --with six --from kaggle==2.2.5 kaggle "$@"; }

# Credentials: either the classic legacy key (~/.kaggle/kaggle.json, or KAGGLE_USERNAME +
# KAGGLE_KEY), or the newer API token (~/.kaggle/access_token, or KAGGLE_API_TOKEN) plus
# KAGGLE_USERNAME (the username is not secret; it names the kernel).
if [ -n "${KAGGLE_USERNAME:-}" ]; then
  KUSER=$KAGGLE_USERNAME
elif [ -s "$HOME/.kaggle/username" ]; then
  KUSER=$(head -n1 "$HOME/.kaggle/username")
elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
  KUSER=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')
else
  echo "no Kaggle username: save ~/.kaggle/kaggle.json, or export KAGGLE_USERNAME=<your username>" >&2
  exit 2
fi
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ ! -f "$HOME/.kaggle/access_token" ] \
   && [ -z "${KAGGLE_KEY:-}" ] && [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "no Kaggle token: see the header of this script" >&2; exit 2
fi
k config view > /dev/null 2>&1 || { echo "Kaggle CLI could not authenticate" >&2; exit 2; }
SLUG="$KUSER/gpulab-nvidia-t4"

if [ "${1:-}" != "--fetch" ]; then
  COMMIT=$(git rev-parse HEAD)
  git fetch -q origin main
  git merge-base --is-ancestor "$COMMIT" origin/main || { echo "push $COMMIT to origin/main first" >&2; exit 2; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "commit your changes first" >&2; exit 2; }
  STAGE=$(mktemp -d)
  sed "s/__COMMIT__/$COMMIT/" deploy/kaggle/run.py > "$STAGE/run.py"
  sed "s/__USERNAME__/$KUSER/" deploy/kaggle/kernel-metadata.template.json > "$STAGE/kernel-metadata.json"
  echo "pushing $SLUG at commit ${COMMIT:0:7} (T4, 3 h cap)"
  k kernels push -p "$STAGE" --accelerator NvidiaTeslaT4 -t 10800
  # poll the remote run (an external queue the harness cannot watch); give up after 4 h
  for _ in $(seq 240); do
    st=$(k kernels status "$SLUG" 2>&1 | tr '[:upper:]' '[:lower:]')
    echo "$(date +%H:%M) $st"
    case "$st" in
      *complete*) break ;;
      *error*|*cancel*) echo "kernel run failed; fetching logs anyway"; break ;;
    esac
    sleep 60
  done
fi

DEST="results/gpu/kaggle-t4-$(date +%Y%m%d-%H%M)"
mkdir -p "$DEST/raw"
k kernels output "$SLUG" -p "$DEST/raw" --force
if [ -f "$DEST/raw/gpulab_results.tar.gz" ]; then
  tar -xzf "$DEST/raw/gpulab_results.tar.gz" -C "$DEST"
  rm "$DEST/raw/gpulab_results.tar.gz"
fi
python3 -c "import json,sys; [print(f\"{s['step']:<18} {s['status']:<8} {s.get('reason','')}\") for s in json.load(open(sys.argv[1]))['steps']]" "$DEST/gpulab_results/session.json" || true
.venv/bin/gpulab analyze
echo "results in $DEST; summary and ANALYSIS.md regenerated"
