import _bootstrap  # noqa: F401
import argparse
from pathlib import Path
from src.common import data_arguments, selected_indices, write_json, protect_dataset
from src.datasets.booster import BoosterDataset
from src.visualization.depth_visualizer import save_panel
from evaluate_zero_shot import load_prediction


def main():
    p = argparse.ArgumentParser(description="Create 6-panel report figures from actual saved predictions")
    data_arguments(p)
    p.set_defaults(max_samples=5)
    p.add_argument("--output-dir", type=Path, default=Path("results/zero_shot_metric_large"))
    p.add_argument("--depth-min", type=float, default=0)
    p.add_argument("--depth-max", type=float, default=5)
    p.add_argument("--error-max", type=float, default=1)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--validity", choices=["all", "consistent"], default="all")
    args = p.parse_args()
    if args.depth_max <= args.depth_min or args.error_max <= 0 or args.dpi < 1:
        p.error("Invalid display ranges or DPI")
    protect_dataset(args.data_root, args.output_dir)
    ds = BoosterDataset(args.data_root, args.split, args.scene, args.subset)
    failures = []
    for i in selected_indices(ds, args.max_samples):
        try:
            s = ds[i]
            raw, out, _ = load_prediction(args.output_dir, ds, i, s)
            save_panel(s, raw, args.output_dir / "figures" / f"{out.name}.png",
                       args.depth_min, args.depth_max, args.error_max, args.dpi, args.validity)
        except Exception as exc:
            failures.append(dict(sample_id=ds.sample_id(i), error=f"{type(exc).__name__}: {exc}"))
            print(f"FAILED {ds.sample_id(i)}: {exc}")
    write_json(args.output_dir / "visualization_failures.json", failures)
    write_json(args.output_dir / "visualization_settings.json", dict(depth_min=args.depth_min, depth_max=args.depth_max,
               error_max=args.error_max, dpi=args.dpi, validity=args.validity,
               alignment="none", depth_units="metres", metric_protocol="calibrated native focal / 300"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
