#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the processed Booster root on this server}"
OUT="${OUTPUT_DIR:-results/zero_shot}"
SPLIT_NAME="${SPLIT:-train}"
mkdir -p logs
python scripts/check_environment.py --strict --import-da3 --output logs/environment.json
python scripts/inspect_booster.py --data-root "$DATA_ROOT" --output-dir logs/dataset_audit
python scripts/run_zero_shot.py --data-root "$DATA_ROOT" --split "$SPLIT_NAME" \
  --output-dir "$OUT" --resume "$@" 2>&1 | tee logs/inference.log
python scripts/evaluate_zero_shot.py --data-root "$DATA_ROOT" --split "$SPLIT_NAME" \
  --output-dir "$OUT" 2>&1 | tee logs/evaluation.log
python scripts/visualize_zero_shot.py --data-root "$DATA_ROOT" --split "$SPLIT_NAME" \
  --output-dir "$OUT" --max-samples 5 2>&1 | tee logs/visualization.log
