#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLEARGRASP_ROOT:?Set CLEARGRASP_ROOT to the uploaded Linux dataset root}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/run_cleargrasp_zero_shot.py \
  --data-root "$CLEARGRASP_ROOT" \
  --checkpoint checkpoints/DA3METRIC-LARGE \
  --split real-test --seed 42 \
  --process-res 560 --device cuda:0 --precision bf16 \
  --output-dir results/cleargrasp_smoke "$@"
