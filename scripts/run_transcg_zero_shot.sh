#!/usr/bin/env bash
set -euo pipefail
# Run inside the existing vggtomast3r environment. No installs or downloads.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/run_transcg_zero_shot.py "$@"
