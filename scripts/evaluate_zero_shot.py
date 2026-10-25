"""Evaluate saved predictions with explicit per-image GT scale alignment."""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path
import numpy as np

from src.common import (data_arguments, selected_indices, sample_output, read_json, write_json,
                        write_csv, write_npy, digest_file, protect_dataset)
from src.datasets.booster import BoosterDataset
from src.evaluation.metrics import MetricAccumulator, align_scale, region_masks


def load_prediction(output_dir, ds, index, sample):
    out = sample_output(output_dir, ds.split, sample["sample_id"])
    meta = read_json(out / "prediction.json")
    if meta["sample_id"] != sample["sample_id"] or meta["split"] != ds.split:
        raise ValueError("Prediction identity mismatch")
    if meta["input_sha256"] != digest_file(ds.paths(index)["rgb"]):
        raise ValueError("Prediction is from a different RGB input")
    if meta["run_config"] != read_json(Path(output_dir) / "run_metadata.json"):
        raise ValueError("Prediction run metadata mismatch")
    if digest_file(out / "raw.npy") != meta["output_sha256"]["raw.npy"]:
        raise ValueError("Prediction checksum mismatch")
    raw = np.load(out / "raw.npy", allow_pickle=False)
    if raw.dtype != np.float32 or raw.shape != sample["depth_gt"].shape:
        raise ValueError("Expected float32 prediction already mapped to saved GT size")
    return raw, out, meta


def evaluate(args):
    protect_dataset(args.data_root, args.output_dir)
    ds = BoosterDataset(args.data_root, args.split, args.scene, args.subset)
    run = read_json(args.output_dir / "run_metadata.json")
    if args.mode == "raw_metric" and run["depth_semantics"] != "metric_model":
        raise ValueError("DA3-BASE/SMALL raw predictions are not metric depth; use scale_aligned")
    accumulators, rows, failures = {}, [], []
    evaluated = 0
    for i in selected_indices(ds, args.max_samples):
        sid = ds.sample_id(i)
        try:
            sample = ds[i]
            raw, out, provenance = load_prediction(args.output_dir, ds, i, sample)
            gt, valid = sample["depth_gt"], sample["valid_mask"].copy()
            if args.validity == "consistent":
                valid &= sample["consistency_mask"]
            if args.mode == "scale_aligned":
                pred, scale = align_scale(raw, gt, valid)
                write_npy(out / "scale_aligned.npy", pred.astype(np.float32))
                write_json(out / "alignment.json", dict(protocol="evaluation-time per-image GT median alignment",
                                                        scale=scale, validity=args.validity,
                                                        raw_sha256=provenance["output_sha256"]["raw.npy"],
                                                        gt_sha256=digest_file(ds.paths(i)["depth_gt"])))
            else:
                pred, scale = raw, 1.0
            for name, mask in region_masks(sample["material_mask"], gt.shape).items():
                mask &= valid
                result = MetricAccumulator().update(pred, gt, mask).result()
                accumulators.setdefault(name, MetricAccumulator()).update(pred, gt, mask)
                eligible = int(mask.sum())
                rows.append(dict(sample_id=sid, region=name, mode=args.mode, scale=scale,
                                 gt_valid_pixels=eligible,
                                 prediction_coverage=result["valid_pixels"]/eligible if eligible else None, **result))
            evaluated += 1
        except Exception as exc:
            failures.append(dict(sample_id=sid, stage="evaluation", error=f"{type(exc).__name__}: {exc}"))
            print(f"FAILED {sid}: {exc}", flush=True)
    regions = {name: a.result() for name, a in accumulators.items()}
    summary = dict(mode=args.mode, aggregation="global_pixel_weighted",
                   alignment="evaluation-time GT median(gt_valid)/median(pred_valid), one scale per image shared across regions" if args.mode == "scale_aligned" else "none",
                   validity=args.validity, selected=len(selected_indices(ds, args.max_samples)),
                   evaluated=evaluated, failed=len(failures),
                   completeness="complete" if not failures else "partial",
                   region_definition=dict(target_classes_2_3=[2, 3], ordinary_class_0=[0], other_classes_0_1=[0, 1]),
                   run_metadata=run, regions=regions)
    write_json(args.output_dir / "metrics_summary.json", summary)
    write_csv(args.output_dir / "metrics_per_image.csv", rows,
              ["sample_id", "region", "mode", "scale", "gt_valid_pixels", "prediction_coverage", "status", "valid_pixels", "samples", "AbsRel", "RMSE", "delta1"])
    write_csv(args.output_dir / "metrics_summary.csv", [dict(region=k, **v) for k, v in regions.items()],
              ["region", "status", "valid_pixels", "samples", "AbsRel", "RMSE", "delta1"])
    inference_failures = read_json(args.output_dir / "inference_failures.json") if (args.output_dir / "inference_failures.json").exists() else []
    write_json(args.output_dir / "failures.json", dict(inference=inference_failures, evaluation=failures))
    print(f"Evaluated {evaluated}/{summary['selected']} images; {len(failures)} failures; {args.mode}")
    return 1 if failures else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    data_arguments(p)
    p.add_argument("--output-dir", type=Path, default=Path("results/zero_shot"))
    p.add_argument("--mode", choices=["scale_aligned", "raw_metric"], default="scale_aligned")
    p.add_argument("--validity", choices=["all", "consistent"], default="all",
                   help="all: finite positive GT AND saved valid; consistent adds stereo consistency mask")
    return p


if __name__ == "__main__":
    raise SystemExit(evaluate(parser().parse_args()))
