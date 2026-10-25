#!/usr/bin/env bash
# Full server run. Reuses the confirmed existing research environment.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the existing processed Booster root on the server}"
OUT="${OUTPUT_DIR:-results/zero_shot_metric_large}"
SPLIT_NAME="${SPLIT:-train}"
GPU_DEVICE="${DEVICE:-cuda:0}"
mkdir -p logs
run_python() { conda run --no-capture-output -n vggtomast3r python "$@"; }
run_python scripts/check_environment.py --strict --import-da3 --device "$GPU_DEVICE" \
  --require-gpu 'RTX 4090D' --output logs/metric_large_environment.json
run_python scripts/inspect_booster.py --data-root "$DATA_ROOT" --output-dir logs/metric_large_audit
run_python scripts/validate_metric_scale.py --data-root "$DATA_ROOT" --split "$SPLIT_NAME" \
  --process-res "${PROCESS_RES:-560}" \
  --output logs/metric_scale_validation.json
checkpoint_args=()
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then checkpoint_args+=(--checkpoint "$CHECKPOINT_DIR"); fi
# Use --max-samples / --scene / --subset here; selections apply to every stage.
run_python scripts/run_zero_shot.py --data-root "$DATA_ROOT" --split "$SPLIT_NAME" \
  --output-dir "$OUT" --model-id depth-anything/DA3METRIC-LARGE \
  --process-res "${PROCESS_RES:-560}" --precision auto --device "$GPU_DEVICE" \
  --require-gpu 'RTX 4090D' --resume "${checkpoint_args[@]}" "$@" 2>&1 | tee logs/metric_large_inference.log
run_python scripts/evaluate_zero_shot.py --data-root "$DATA_ROOT" --split "$SPLIT_NAME" \
  --output-dir "$OUT" --mode raw_metric "$@" 2>&1 | tee logs/metric_large_evaluation.log
run_python scripts/visualize_zero_shot.py --data-root "$DATA_ROOT" --split "$SPLIT_NAME" \
  --output-dir "$OUT" "$@" 2>&1 | tee logs/metric_large_visualization.log
