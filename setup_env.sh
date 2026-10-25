#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python - <<'PY'
import sys
if sys.version_info[:3] != (3, 10, 20):
    raise SystemExit('Activate an isolated Python 3.10.20 environment first (environment.yml).')
PY
command -v nvidia-smi >/dev/null && nvidia-smi || true
python scripts/check_environment.py --output logs/environment_before.json
# No driver/toolkit/system library changes. Binary wheels only for torch stack.
python -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cu121 \
  -c constraints.txt 'torch==2.3.1+cu121' 'torchvision==0.18.1+cu121'
# Manylinux2014/2_17 wheels are chosen by pip on GLIBC 2.17. Small pure-Python
# dependencies (e.g. antlr runtime) may use their ordinary source distribution.
python -m pip install --only-binary=:all: --no-binary=antlr4-python3-runtime -c constraints.txt -r requirements.txt
DA3_COMMIT=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
DA3_DIR="$(pwd)/vendor/Depth-Anything-3"
if [[ ! -d "$DA3_DIR/.git" ]]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DA3_DIR"
fi
if [[ -n "$(git -C "$DA3_DIR" status --porcelain)" ]]; then
  # Only accept the exact patch our installer applied; reject unrelated edits.
  python scripts/patch_da3_exports.py "$DA3_DIR" --verify
else
  git -C "$DA3_DIR" checkout --detach "$DA3_COMMIT"
  python scripts/patch_da3_exports.py "$DA3_DIR"
fi
# Source-only .pth registration avoids build isolation and all upstream dependency
# resolution. requirements.txt covers the actual import graph used by this baseline.
python - "$DA3_DIR/src" <<'PY'
import site, sys
from pathlib import Path
Path(site.getsitepackages()[0], 'da3_booster_source.pth').write_text(str(Path(sys.argv[1]).resolve())+'\n')
PY
python scripts/check_environment.py --strict --import-da3 --output logs/environment_after.json
python -m pytest -q
python -m pip freeze > logs/requirements_resolved.txt
git -C "$DA3_DIR" diff > logs/da3_compatibility.patch
echo 'Environment prepared. No checkpoint has been downloaded; no inference has run.'
