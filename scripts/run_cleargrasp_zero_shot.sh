#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# All data/checkpoint/output paths are read from the selected YAML.
# Example: bash scripts/run_cleargrasp_zero_shot.sh --config configs/cleargrasp_zero_shot.yaml
python scripts/run_cleargrasp_zero_shot.py "$@"
