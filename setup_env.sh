#!/usr/bin/env bash
# Compatibility name retained. This script only CHECKS the existing environment.
set -euo pipefail
cd "$(dirname "$0")"
conda run --no-capture-output -n vggtomast3r \
  python scripts/check_environment.py --strict --import-da3 \
  --require-gpu 'RTX 4090D' --output logs/metric_large_environment.json
echo 'Read-only preflight passed. No environment/dependency/source/checkpoint was changed.'
