#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Server only. Uses the existing environment; never creates/updates an environment.
# All data/checkpoint/output paths are read from YAML, not environment variables.
# Example: bash scripts/run_cleargrasp_zero_shot.sh
# Optional: append --config configs/your_cleargrasp_run.yaml
if ! command -v conda >/dev/null 2>&1; then
  echo 'ERROR: conda is not available on PATH. Use the server shell with the existing Conda installation.' >&2
  exit 1
fi
echo 'ClearGrasp: random 20 samples -> annotation preflight -> DA3 -> evaluation -> visualization'
echo 'Using existing Conda environment: vggtomast3r; all paths and seed come from YAML.'
conda run --no-capture-output -n vggtomast3r \
  python -u scripts/run_cleargrasp_zero_shot.py --smoke-test "$@"
