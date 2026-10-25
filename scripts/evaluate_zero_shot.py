"""Raw Metric Evaluation. GT is used only for errors/validity, never scaling."""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path
import numpy as np

from src.common import (data_arguments, selected_indices, sample_output, read_json, write_json,
                        write_csv, digest_file, protect_dataset)
from src.datasets.booster import BoosterDataset
from src.evaluation.metrics import MetricAccumulator, region_masks
from src.inference.da3_zero_shot import MODEL_ID, METRIC_PROTOCOL, metric_depth_from_canonical, resize_prediction


def load_prediction(output_dir, ds, index, sample):
    out = sample_output(output_dir, ds.split, sample["sample_id"])
    meta = read_json(out / "prediction.json")
    if meta["sample_id"] != sample["sample_id"] or meta["split"] != ds.split:
        raise ValueError("Prediction identity mismatch")
    if meta["input_sha256"] != digest_file(ds.paths(index)["rgb"]):
        raise ValueError("Prediction is from a different RGB input")
    if meta["run_config"] != read_json(Path(output_dir) / "run_metadata.json"):
        raise ValueError("Prediction run metadata mismatch")
    if (meta["run_config"].get("model_id") != MODEL_ID or
            meta["run_config"].get("metric_protocol") != METRIC_PROTOCOL or
            meta["run_config"].get("gt_alignment") is not False or meta.get("raw_units") != "metres"):
        raise ValueError("Prediction is not a verified unaligned Metric-Large output")
    if meta.get("calibration_sha256") != digest_file(ds.paths(index)["meta"]):
        raise ValueError("Prediction calibration metadata has changed")
    # Verify canonical->metric provenance before accepting metre-labelled arrays.
    for name in ["canonical_native.npy", "raw_native.npy"]:
        if digest_file(out / name) != meta["output_sha256"][name]:
            raise ValueError("Metric conversion source checksum mismatch")
    canonical = np.load(out / "canonical_native.npy", allow_pickle=False)
    native = np.load(out / "raw_native.npy", allow_pickle=False)
    expected, conversion = metric_depth_from_canonical(canonical, sample["updated_K"], sample["rgb"].shape[:2])
    if conversion != meta["metric_conversion"] or not np.array_equal(native, expected, equal_nan=True):
        raise ValueError("Metric conversion provenance mismatch (possible missing/double scaling)")
    if digest_file(out / "raw.npy") != meta["output_sha256"]["raw.npy"]:
        raise ValueError("Prediction checksum mismatch")
    raw = np.load(out / "raw.npy", allow_pickle=False)
    if raw.dtype != np.float32 or raw.shape != sample["depth_gt"].shape:
        raise ValueError("Expected float32 prediction already mapped to saved GT size")
    if not np.array_equal(raw, resize_prediction(native, raw.shape), equal_nan=True):
        raise ValueError("Raw metric array does not match converted native depth on the GT grid")
    return raw, out, meta


def evaluate(args):
    protect_dataset(args.data_root, args.output_dir)
    ds = BoosterDataset(args.data_root, args.split, args.scene, args.subset)
    run = read_json(args.output_dir / "run_metadata.json")
    if (run.get("model_id") != MODEL_ID or run.get("depth_semantics") != "raw_metric_metres" or
            run.get("metric_protocol") != METRIC_PROTOCOL or run.get("gt_alignment") is not False):
        raise ValueError("Raw metric evaluation requires the calibrated DA3METRIC-LARGE protocol")
    accumulators, rows, failures = {}, [], []
    evaluated = 0
    for i in selected_indices(ds, args.max_samples):
        sid = ds.sample_id(i)
        try:
            sample = ds[i]
            raw, _, _ = load_prediction(args.output_dir, ds, i, sample)
            gt, valid = sample["depth_gt"], sample["valid_mask"].copy()
            if args.validity == "consistent":
                valid &= sample["consistency_mask"]
            pred = raw
            for name, mask in region_masks(sample["material_mask"], gt.shape).items():
                mask &= valid
                result = MetricAccumulator().update(pred, gt, mask).result()
                accumulators.setdefault(name, MetricAccumulator()).update(pred, gt, mask)
                eligible = int(mask.sum())
                rows.append(dict(sample_id=sid, region=name, mode=args.mode, gt_alignment=False,
                                 gt_valid_pixels=eligible,
                                 prediction_coverage=result["valid_pixels"]/eligible if eligible else None, **result))
            evaluated += 1
        except Exception as exc:
            failures.append(dict(sample_id=sid, stage="evaluation", error=f"{type(exc).__name__}: {exc}"))
            print(f"FAILED {sid}: {exc}", flush=True)
    regions = {name: a.result() for name, a in accumulators.items()}
    summary = dict(mode=args.mode, aggregation="global_pixel_weighted",
                   alignment="none", gt_alignment=False,
                   metric_conversion="calibrated native focal / 300; independent of GT",
                   validity=args.validity, selected=len(selected_indices(ds, args.max_samples)),
                   evaluated=evaluated, failed=len(failures),
                   completeness="complete" if not failures else "partial",
                   region_definition=dict(target_classes_2_3=[2, 3], ordinary_class_0=[0], other_classes_0_1=[0, 1]),
                   run_metadata=run, regions=regions)
    write_json(args.output_dir / "metrics_summary.json", summary)
    write_csv(args.output_dir / "metrics_per_image.csv", rows,
              ["sample_id", "region", "mode", "gt_alignment", "gt_valid_pixels", "prediction_coverage", "status", "valid_pixels", "samples", "AbsRel", "RMSE", "delta1"])
    write_csv(args.output_dir / "metrics_summary.csv", [dict(region=k, **v) for k, v in regions.items()],
              ["region", "status", "valid_pixels", "samples", "AbsRel", "RMSE", "delta1"])
    inference_failures = read_json(args.output_dir / "inference_failures.json") if (args.output_dir / "inference_failures.json").exists() else []
    write_json(args.output_dir / "failures.json", dict(inference=inference_failures, evaluation=failures))
    print(f"Evaluated {evaluated}/{summary['selected']} images; {len(failures)} failures; {args.mode}")
    return 1 if failures else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    data_arguments(p)
    p.add_argument("--output-dir", type=Path, default=Path("results/zero_shot_metric_large"))
    p.add_argument("--mode", choices=["raw_metric"], default="raw_metric")
    p.add_argument("--validity", choices=["all", "consistent"], default="all",
                   help="all: finite positive GT AND saved valid; consistent adds stereo consistency mask")
    return p


if __name__ == "__main__":
    raise SystemExit(evaluate(parser().parse_args()))
