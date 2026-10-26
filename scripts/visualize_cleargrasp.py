import argparse
import _bootstrap
from src.visualization.cleargrasp import visualize_saved

def main():
    p = argparse.ArgumentParser(description='Render ClearGrasp saved predictions on CPU')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--data-root')
    p.add_argument('--depth-max', type=float, default=2.)
    p.add_argument('--error-max', type=float, default=0.5)
    args = vars(p.parse_args())
    failures = visualize_saved(args.pop('output_dir'), **args)
    print('Visualization failures:', len(failures))
    return bool(failures)

if __name__ == '__main__':
    raise SystemExit(main())
