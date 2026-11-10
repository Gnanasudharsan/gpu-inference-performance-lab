#!/usr/bin/env bash
# Build the static dashboard from results/summary.json and deploy it to Vercel production.
# The staging directory is named after the Vercel project so the CLI links to it.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
STAGE=$(mktemp -d)/gpu-inference-performance-lab
"$ROOT/.venv/bin/gpulab" analyze --results "$ROOT/results" --summary "$ROOT/results/summary.json" \
  --report "$ROOT/results/ANALYSIS.md" > /dev/null
(cd "$ROOT/web" && npm run build > /dev/null)
mkdir -p "$STAGE"
cp -R "$ROOT/web/out/." "$STAGE/"
cp "$ROOT/deploy/vercel.json" "$STAGE/vercel.json"
(cd "$STAGE" && vercel deploy --prod --yes)
URL=https://gpu-inference-performance-lab.vercel.app
for p in / /live/ /data/summary.json; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "$URL$p")
  echo "$p -> $code"; [ "$code" = 200 ] || { echo "verification failed"; exit 1; }
done
