"""CPU-only shared-support paired analysis."""
import _bootstrap
from src.cleargrasp_paired_config import parse_args, print_config
from src.evaluation.cleargrasp_paired import evaluate

if __name__ == '__main__':
    args = parse_args()
    if args.print_config:
        print_config(args)
    else:
        raise SystemExit(evaluate(args))
