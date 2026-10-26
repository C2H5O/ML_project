"""Infer only manifest-matched opaque RGB using the existing DA3 wrapper."""
import _bootstrap
from src.cleargrasp_paired_config import parse_args, print_config
from src.cleargrasp_paired import run

if __name__ == '__main__':
    args = parse_args()
    if args.print_config:
        print_config(args)
    else:
        raise SystemExit(run(args))
