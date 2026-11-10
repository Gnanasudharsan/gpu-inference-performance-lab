#!/usr/bin/env bash
# Run the NVIDIA T4 measurement session on Kaggle (free tier) and ingest the results.
#
# Prerequisites (one-time, done by the account owner):
#   1. Kaggle account with phone verification (needed for internet + GPU in kernels).
#   2. Authenticate the CLI. Easiest: `uvx --with six --from kaggle==2.2.5 kaggle auth login`
#      (browser sign-in; nothing to paste). Alternatives: an API token in ~/.kaggle/access_token
#      or a legacy ~/.kaggle/kaggle.json. Put your username in ~/.kaggle/username (not secret).
#   3. The commit to measure is pushed to GitHub (the kernel clones it by SHA).
#
# Usage: scripts/kaggle_run.sh                # full session: push, run (<= 3 h), wait, download, analyze
#        scripts/kaggle_run.sh --task diag    # CUDA-graph memory-safety diagnostic (<= 30 min)
#        scripts/kaggle_run.sh --fetch        # only download + ingest the latest run
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
# Works with any credential the CLI supports (OAuth via `kaggle auth login`, an API token, or
# a legacy key): ask the API itself whether we are authenticated.
k kernels list --mine > /dev/null 2>&1 || {
  echo "Kaggle CLI is not authenticated: run 'uvx --with six --from kaggle==2.2.5 kaggle auth login'" >&2
  exit 2
}
SLUG="$KUSER/gpulab-nvidia-t4"

TASK=session
TIMEOUT=10800
DEST="results/gpu/kaggle-t4-$(date +%Y%m%d-%H%M)"
case "${1:-}" in
  --task) TASK=${2:?session or diag}; shift 2 ;;
esac
if [ "$TASK" = diag ]; then TIMEOUT=1800; DEST="results/gpu/diag-t4-$(date +%Y%m%d-%H%M)"; fi

if [ "${1:-}" != "--fetch" ]; then
  COMMIT=$(git rev-parse HEAD)
  git fetch -q origin main
  git merge-base --is-ancestor "$COMMIT" origin/main || { echo "push $COMMIT to origin/main first" >&2; exit 2; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "commit your changes first" >&2; exit 2; }
  STAGE=$(mktemp -d)
  sed -e "s/__COMMIT__/$COMMIT/" -e "s/__TASK__/$TASK/" deploy/kaggle/run.py > "$STAGE/run.py"
  sed "s/__USERNAME__/$KUSER/" deploy/kaggle/kernel-metadata.template.json > "$STAGE/kernel-metadata.json"
  echo "pushing $SLUG at commit ${COMMIT:0:7} task=$TASK (T4, ${TIMEOUT}s cap)"
  k kernels push -p "$STAGE" --accelerator NvidiaTeslaT4 -t "$TIMEOUT"
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

mkdir -p "$DEST/raw"
k kernels output "$SLUG" -p "$DEST/raw" --force
if [ -f "$DEST/raw/gpulab_results.tar.gz" ]; then
  tar -xzf "$DEST/raw/gpulab_results.tar.gz" -C "$DEST"
  rm "$DEST/raw/gpulab_results.tar.gz"
fi
if [ "$TASK" = diag ]; then
  mv "$DEST/raw/gpulab_diag" "$DEST/" 2>/dev/null || true
  cat "$DEST/gpulab_diag/diag_cuda_graph.json" 2>/dev/null || echo "no diagnostic output (see $DEST/raw/*.log)"
  tail -3 "$DEST/gpulab_diag/pytest_gpu.log" 2>/dev/null || true
  exit 0
fi
python3 -c "import json,sys; [print(f\"{s['step']:<18} {s['status']:<8} {s.get('reason','')}\") for s in json.load(open(sys.argv[1]))['steps']]" "$DEST/gpulab_results/session.json" || true
.venv/bin/gpulab analyze
echo "results in $DEST; summary and ANALYSIS.md regenerated"
