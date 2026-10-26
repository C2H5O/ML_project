import argparse
import _bootstrap
from src.evaluation.cleargrasp import evaluate_saved

def protocol_arguments(p):
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pair-draws', type=int, default=20000)
    p.add_argument('--gt-gap-m', type=float, default=0.02)
    p.add_argument('--ring-radius', type=int, default=30)
    p.add_argument('--boundary-margin', type=int, default=2)
    p.add_argument('--compression-ratio', type=float, default=0.25)

def main():
    p = argparse.ArgumentParser(description='CPU evaluation of saved ClearGrasp predictions')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--data-root', help='Override original root after moving data to another host')
    protocol_arguments(p)
    a = vars(p.parse_args())
    failures = evaluate_saved(a.pop('output_dir'), **a)
    print('Evaluation failures:', len(failures))
    return bool(failures)

if __name__ == '__main__':
    raise SystemExit(main())
