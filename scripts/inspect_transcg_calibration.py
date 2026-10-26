"""CPU-only, read-only camera-info and original manifest/RGB inspection."""
import _bootstrap  # noqa: F401
import json
import sys
from pathlib import Path

from src.transcg_config import TransCGArgumentParser
from src.transcg_calibration import inspect_calibration


def parser():
    p = TransCGArgumentParser(description=__doc__)
    p.add_argument("--calibration-root", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, default=Path("results/transcg_zero_shot"))
    p.add_argument("--intrinsics", type=Path)
    p.add_argument("--intrinsics-shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--camera", type=int, choices=[1, 2], default=1)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    report = inspect_calibration(args.calibration_root, args.camera, args.intrinsics,
                                 args.intrinsics_shape, args.input_dir)
    print(json.dumps(report, indent=2, allow_nan=False))
    print("K shape/values and observed RGB size do not prove K's calibration grid.", file=sys.stderr, flush=True)
    print("Confirm the camera and original grid from the calibration provenance, then set --intrinsics-shape.", file=sys.stderr, flush=True)
    raise SystemExit(0 if report["ready_for_metric"] else 1)
