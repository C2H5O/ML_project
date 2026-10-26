"""Calibrate saved D435 canonical predictions on CPU. Never imports torch/DA3."""
import _bootstrap  # noqa: F401
import argparse
import json
import math
from pathlib import Path
import shutil
import time

from src.common import (atomic_write, digest_file, write_json, write_npy,
                        write_csv, protect_dataset, positive_int)
from src.datasets.transcg import (load_calibration, calibrated_prediction, prediction_dir,
                                  read_annotations)
from src.inference.da3_zero_shot import resize_prediction
from src.transcg_calibration import validate_source_run, validate_source_sample
from src.transcg_config import TransCGArgumentParser
from src.evaluation.transcg import TransCGMetrics
from src.visualization.transcg import regenerate_figures


def parser():
    p = TransCGArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("results/transcg_zero_shot"))
    # Deliberately a different config key from inference output_dir. Both CLI
    # spellings point at the calibrated result; YAML output_dir is ignored here.
    p.add_argument("--output-dir", "--calibrated-output-dir", dest="calibrated_output_dir", type=Path,
                   default=Path("results/transcg_metric_calibrated"))
    p.add_argument("--calibration-root", type=Path)
    p.add_argument("--intrinsics", type=Path)
    p.add_argument("--intrinsics-shape", nargs=2, type=positive_int, metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--camera", type=int, choices=[1], default=1)
    p.add_argument("--checkpoint", type=Path, help="Optional streaming hash verification; no weights are deserialized")
    p.add_argument("--depth-max", type=float, default=2.0)
    p.add_argument("--error-max", type=float, default=1.0)
    p.add_argument("--zoom-regions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print-config", action="store_true")
    return p


def copy_read_only(source, target):
    # Copies, not hardlinks: later editing derived files cannot alter originals.
    def writer(stream):
        with Path(source).open("rb") as original:
            shutil.copyfileobj(original, stream)
    atomic_write(target, writer)


def run(args):
    source, output = args.input_dir.resolve(), args.calibrated_output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Calibrated output must be disjoint from the original zero-shot directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Calibrated output is nonempty; choose a fresh directory. No overwrite/reconversion is allowed")
    if not all(math.isfinite(v) and v > 0 for v in (args.depth_max, args.error_max)):
        raise ValueError("Colour limits must be finite positive values")
    manifest, source_meta, checkpoint_report = validate_source_run(source, args.checkpoint)
    original_config = source_meta["run_config"]
    # RGB locations always come from the saved manifest, never from calibration_root.
    data_root = Path(manifest["selection"]["root"])
    protect_dataset(data_root, output)
    if args.calibration_root is not None:
        protect_dataset(args.calibration_root, output)
    calibration = load_calibration(data_root, 1, args.intrinsics, args.intrinsics_shape, args.calibration_root)
    if calibration is None:
        raise ValueError("Reliable D435 intrinsics AND confirmed original RGB grid are required; run inspect_transcg_calibration.py. No metric output was generated")
    source_hashes = {name: digest_file(source / name) for name in ("sample_manifest.json", "run_metadata.json")}
    config = dict(**original_config)
    config.update(calibration=calibration, metric_units="metres", gt_alignment=False,
                  postprocess="CPU canonical-to-metric exactly once", source_run=str(source))
    copy_read_only(source / "sample_manifest.json", output / "sample_manifest.json")
    copy_read_only(source / "run_metadata.json", output / "source_run_metadata.json")
    metadata = dict(run_config=config, model=source_meta["model"], operation="CPU calibration of saved network output",
        model_called=False, gpu_used=False, gt_alignment=False, calibration=calibration,
        checkpoint_verification=checkpoint_report, source_run=str(source), source_sha256=source_hashes,
        status="running")
    write_json(output / "run_metadata.json", metadata)
    failures, records = [], []
    metrics = TransCGMetrics()
    for sample in manifest["samples"]:
        folder = prediction_dir(output, sample)
        try:
            start = time.perf_counter()
            canonical, saved, canonical_path, original_prediction = validate_source_sample(source, sample, original_config)
            # No GT is read until AFTER this GT-independent conversion.
            metric, conversion = calibrated_prediction(canonical, sample["input_shape"], calibration)
            copy_read_only(canonical_path, folder / "canonical_depth.npy")
            copy_read_only(original_prediction, folder / "source_prediction.json")
            write_npy(folder / "metric_depth.npy", metric)
            conversion_seconds = time.perf_counter() - start
            gt, mask, annotations = read_annotations(sample, sample["input_shape"])
            score = metrics.update(sample["sample_id"], resize_prediction(metric, sample["input_shape"]), gt, mask, annotations)
            record = dict(**sample, run_config=config, status="success", error=None,
                canonical_units=original_config["canonical_units"], metric_units="metres", native_shape=list(metric.shape),
                metric_conversion=conversion, calibration_source=calibration, annotations=annotations,
                gt_alignment=False, postprocess_seconds=conversion_seconds,
                seconds=saved["seconds"], peak_cuda_allocated_bytes=saved.get("peak_cuda_allocated_bytes"),
                timing_note="seconds/memory copied from original inference; postprocess_seconds is CPU validation/conversion/copy",
                prediction_is_metric=saved.get("prediction_is_metric"),
                source_prediction=dict(path=str(original_prediction), sha256=digest_file(original_prediction),
                                       canonical_path=str(canonical_path), canonical_sha256=digest_file(canonical_path)),
                output_sha256={n: digest_file(folder / n) for n in ("canonical_depth.npy", "metric_depth.npy")},
                raw_metric_evaluation=score)
            write_json(folder / "prediction.json", record)
            records.append(dict(sample_id=sample["sample_id"], status="success", metric_conversion=conversion,
                                postprocess_seconds=conversion_seconds))
            print(f"CALIBRATED {sample['sample_id']} focal_native={conversion['focal_native_px']:.6g}px multiplier={conversion['multiplier']:.6g}", flush=True)
        except Exception as exc:
            failure = dict(sample_id=sample["sample_id"], path=sample["path"], status="failed",
                           stage="cpu_calibration", error=f"{type(exc).__name__}: {exc}")
            failures.append(failure)
            records.append(failure)
            write_json(folder / "failure.json", failure)
            print(f"FAILED {sample['sample_id']}: {exc}", flush=True)
        finally:
            write_json(output / "inference_failures.json", failures)
            write_json(output / "inference_summary.json", dict(operation="CPU calibration; no inference executed",
                selected=len(manifest["samples"]), records=records, failed=len(failures),
                pending=len(manifest["samples"]) - len(records)))
    report = metrics.result()
    if report["regions"]["full_image"]["valid_pixels"]:
        write_json(output / "metrics_summary.json", report)
        fields = ["Region", "status", "AbsRel", "RMSE", "delta1", "valid_pixels", "samples"]
        write_csv(output / "metrics_summary.csv", [dict(Region=name, **scores) for name, scores in report["regions"].items()], fields)
    else:
        write_json(output / "evaluation_status.json", dict(status="unavailable", **report))
    figure_failures = regenerate_figures(output, args.depth_max, args.error_max, args.zoom_regions)
    unchanged = all(digest_file(source / name) == value for name, value in source_hashes.items())
    if not unchanged:
        failures.append(dict(stage="source_integrity", status="failed", error="Original source metadata changed during CPU calibration"))
    metadata.update(status="completed_with_failures" if failures or figure_failures else "completed",
        source_metadata_unchanged=unchanged, evaluation_available=bool(report["regions"]["full_image"]["valid_pixels"]),
        visualization_failures=figure_failures, evaluation_skipped=report["skipped"])
    write_json(output / "run_metadata.json", metadata)
    write_json(output / "inference_failures.json", failures)
    return 1 if failures or figure_failures else 0


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if args.print_config:
        print(json.dumps(vars(args), indent=2, default=str))
    else:
        try:
            raise SystemExit(run(args))
        except (ValueError, OSError, KeyError) as exc:
            p.exit(1, f"CPU calibration failed: {exc}\n")
