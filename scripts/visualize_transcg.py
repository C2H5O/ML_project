"""Regenerate TransCG figures/contact sheet from saved arrays, without inference."""
import _bootstrap  # noqa: F401
from pathlib import Path
from src.visualization.transcg import regenerate_figures
from src.transcg_config import TransCGArgumentParser


if __name__ == "__main__":
    p = TransCGArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/transcg_zero_shot"))
    p.add_argument("--depth-max", type=float, default=2.0)
    p.add_argument("--error-max", type=float, default=1.0)
    args = p.parse_args()
    if args.depth_max <= 0 or args.error_max <= 0:
        p.error("Colour limits must be positive")
    raise SystemExit(1 if regenerate_figures(args.output_dir, args.depth_max, args.error_max) else 0)
