import _bootstrap
from src.evaluation.cleargrasp import evaluate_saved
from src.cleargrasp_config import ClearGraspArgumentParser, print_config

def protocol_arguments(p):
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pair-draws', type=int, default=20000)
    p.add_argument('--gt-gap-m', type=float, default=0.02)
    p.add_argument('--ring-radius', type=int, default=30)
    p.add_argument('--boundary-margin', type=int, default=2)
    p.add_argument('--compression-ratio', type=float, default=0.25)

def main():
    p = ClearGraspArgumentParser(description='CPU evaluation of saved ClearGrasp predictions')
    protocol_arguments(p)
    a = p.parse_args()
    if a.print_config:
        print_config(a)
        return 0
    failures = evaluate_saved(a.output_dir, data_root=a.data_root, seed=a.seed,
        pair_draws=a.pair_draws, gt_gap_m=a.gt_gap_m, ring_radius=a.ring_radius,
        boundary_margin=a.boundary_margin, compression_ratio=a.compression_ratio)
    print('Evaluation failures:', len(failures))
    return bool(failures)

if __name__ == '__main__':
    raise SystemExit(main())
