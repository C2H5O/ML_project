"""CPU-only fixed-range paired panels and contact sheet."""
import _bootstrap
from src.cleargrasp_paired_config import parse_args, print_config
from src.visualization.cleargrasp_paired import visualize

if __name__ == '__main__':
    args = parse_args()
    if args.print_config:
        print_config(args)
    else:
        raise SystemExit(visualize(args))
