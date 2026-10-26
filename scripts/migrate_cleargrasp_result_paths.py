"""Preview or explicitly migrate saved ClearGrasp prediction/figure directories."""
import json
import _bootstrap
from src.cleargrasp_config import ClearGraspArgumentParser, print_config
from src.cleargrasp_migration import migrate_results
from src.common import positive_int, protect_dataset


def main():
    p = ClearGraspArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument('--dry-run', action='store_true', help='Default: validate and print mapping; write nothing')
    actions.add_argument('--apply', action='store_true', help='Validate every saved sample, then move directories')
    actions.add_argument('--recover', action='store_true', help='Roll back an interrupted migration using its journal')
    p.add_argument('--expected-samples', dest='analysis_expected_samples', type=positive_int, default=20)
    args = p.parse_args()
    if args.print_config:
        print_config(args)
        return 0
    protect_dataset(args.data_root, args.output_dir)
    plan = migrate_results(args.output_dir, args.analysis_expected_samples, args.apply, args.recover)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
