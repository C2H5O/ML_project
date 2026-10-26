import _bootstrap
from src.visualization.cleargrasp import visualize_saved
from src.cleargrasp_config import ClearGraspArgumentParser, print_config

def main():
    p = ClearGraspArgumentParser(description='Render ClearGrasp saved predictions on CPU')
    p.add_argument('--depth-max', type=float, default=2.)
    p.add_argument('--error-max', type=float, default=0.5)
    args = p.parse_args()
    if args.print_config:
        print_config(args)
        return 0
    failures = visualize_saved(args.output_dir, data_root=args.data_root,
                               depth_max=args.depth_max, error_max=args.error_max)
    print('Visualization failures:', len(failures))
    return bool(failures)

if __name__ == '__main__':
    raise SystemExit(main())
