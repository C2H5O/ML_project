"""CPU-only geometry analysis of every sample in the saved ClearGrasp manifest."""
import _bootstrap
from src.cleargrasp_config import ClearGraspArgumentParser, print_config
from src.common import positive_int
from src.evaluation.cleargrasp_analysis import analyze_saved_geometry
from evaluate_cleargrasp import protocol_arguments


def main():
    p = ClearGraspArgumentParser(description=__doc__)
    protocol_arguments(p)
    p.add_argument('--split', default='real-test')
    p.add_argument('--expected-samples', dest='analysis_expected_samples', type=positive_int, default=20)
    p.add_argument('--min-pairs', dest='analysis_min_pairs', type=positive_int, default=50)
    p.add_argument('--min-unique-pixels', dest='analysis_min_unique_pixels', type=positive_int, default=10)
    p.add_argument('--boundary-width', dest='analysis_boundary_width', type=positive_int, default=4)
    p.add_argument('--min-region-pixels', dest='analysis_min_region_pixels', type=positive_int, default=10)
    p.add_argument('--baseline-rtol', dest='analysis_baseline_rtol', type=float, default=1e-6)
    p.add_argument('--baseline-atol', dest='analysis_baseline_atol', type=float, default=1e-8)
    p.add_argument('--depth-max', type=float, default=2.)
    p.add_argument('--error-max', type=float, default=.5)
    a = p.parse_args()
    if a.print_config:
        print_config(a)
        return 0
    summary = analyze_saved_geometry(a.output_dir, a.data_root, expected_samples=a.analysis_expected_samples,
        split=a.split, seed=a.seed, pair_draws=a.pair_draws, gt_gap_m=a.gt_gap_m,
        ring_radius=a.ring_radius, boundary_margin=a.boundary_margin, compression_ratio=a.compression_ratio,
        min_pairs=a.analysis_min_pairs, min_unique_pixels=a.analysis_min_unique_pixels,
        boundary_width=a.analysis_boundary_width, min_region_pixels=a.analysis_min_region_pixels,
        depth_max=a.depth_max, error_max=a.error_max,
        baseline_rtol=a.analysis_baseline_rtol, baseline_atol=a.analysis_baseline_atol)
    print(f"Analysis status: {summary['status']}; read {summary['verified_samples']}/{summary['selected_samples']} predictions")
    print('Report:', a.output_dir/'geometry_analysis'/'report.md')
    return 0 if summary['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
