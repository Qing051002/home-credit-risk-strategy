#!/usr/bin/env bash
# Execute all notebooks in order, saving outputs in place.
# Usage:  bash tools/run_pipeline.sh            (real data under data/ or $HCRISK_DATA_DIR)
#         bash tools/run_pipeline.sh 04 06      (run notebooks 04 to 06 only)
set -euo pipefail
cd "$(dirname "$0")/../notebooks"
FROM=${1:-01}; TO=${2:-10}
for nb in [0-9][0-9]_*.ipynb; do
  n=${nb:0:2}
  if [[ "$n" < "$FROM" || "$n" > "$TO" ]]; then continue; fi
  echo "=== $nb ==="
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb"
done
