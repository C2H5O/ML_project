"""Independent, offline TransCG RGB-only DA3METRIC-LARGE inference."""
import _bootstrap  # noqa: F401
import argparse
import json
import os
from pathlib import Path
import random

import numpy as np

from src.common import (read_json, write_json, write_npy, digest_file, positive_int,
                        protect_dataset)
from src.datasets.transcg import (scan_rgb, discover_root, sample_rgb, read_rgb,
    load_calibration, calibrated_prediction, prediction_dir, read_annotations)
from src.inference.da3_zero_shot import DA3Predictor, resize_prediction
from src.evaluation.metrics import MetricAccumulator
from src.visualization.transcg import regenerate_figures
from src.transcg_config import TransCGArgumentParser


def parser():
    p = TransCGArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/DA3METRIC-LARGE"))
    p.add_argument("--scene-start", type=int, default=1)
    p.add_argument("--scene-end", type=int, default=10)
    p.add_argument("--camera", type=int, choices=[1, 2], default=1)
    p.add_argument("--num-samples", type=positive_int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling", choices=["global", "per_scene"], default="global")
    p.add_argument("--process-res", type=positive_int, default=560)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=["bf16", "auto", "fp16", "fp32"], default="bf16")
    p.add_argument("--output-dir", type=Path, default=Path("results/transcg_zero_shot"))
    p.add_argument("--intrinsics", type=Path, help="Trusted camera-specific local 3x3 .npy; never inferred from GT")
    p.add_argument("--intrinsics-shape", nargs=2, type=positive_int, metavar=("HEIGHT", "WIDTH"),
                   help="Confirmed pixel grid of K, must match original RGB dimensions")
    p.add_argument("--depth-max", type=float, default=2.0, help="Shared GT/metric colour maximum in metres")
    p.add_argument("--error-max", type=float, default=1.0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--print-config", action="store_true", help="Print resolved arguments and exit without scanning data or loading a model")
    return p


def validate_saved(folder, sample, config):
    record = read_json(folder / "prediction.json")
    if (record["status"] != "success" or record["run_config"] != config or
            record["sample_id"] != sample["sample_id"] or
            record["input_sha256"] != digest_file(sample["path"])):
        raise ValueError("Saved prediction input/config mismatch; use a fresh output directory")
    expected = {"canonical_depth.npy"} | ({"metric_depth.npy"} if config["calibration"] else set())
    if set(record["output_sha256"]) != expected:
        raise ValueError("Saved prediction units/files mismatch")
    for name, checksum in record["output_sha256"].items():
        array = np.load(folder / name, allow_pickle=False)
        if (array.dtype != np.float32 or list(array.shape) != record["native_shape"] or
                digest_file(folder / name) != checksum):
            raise ValueError(f"Corrupt resumed prediction: {name}")
    return record


def evaluate_saved(output, samples, records, calibrated):
    # This directory has already passed the TransCG ownership/config checks.
    # A resumed run must not retain obsolete scores for newly invalid samples.
    (output / "metrics_summary.json").unlink(missing_ok=True)
    if not calibrated:
        return
    full, transparent = MetricAccumulator(), MetricAccumulator()
    per_image, skipped = [], []
    successful = {r["sample_id"] for r in records if r["status"] in {"success", "resumed"}}
    for sample in samples:
        if sample["sample_id"] not in successful:
            continue
        try:
            folder = prediction_dir(output, sample)
            gt, mask, info = read_annotations(sample, sample["input_shape"])
            if gt is None:
                skipped.append(dict(sample_id=sample["sample_id"], reason=info))
                continue
            pred = resize_prediction(np.load(folder / "metric_depth.npy", allow_pickle=False), gt.shape)
            result = MetricAccumulator().update(pred, gt).result()
            full.update(pred, gt)
            if mask is not None:
                transparent.update(pred, gt, mask)
            valid_gt = np.isfinite(gt) & (gt > 0)
            per_image.append(dict(sample_id=sample["sample_id"], **result,
                valid_gt_pixels=int(valid_gt.sum()),
                prediction_coverage=result["valid_pixels"] / int(valid_gt.sum()) if valid_gt.any() else None,
                annotations=info))
        except Exception as exc:
            skipped.append(dict(sample_id=sample["sample_id"], reason=f"{type(exc).__name__}: {exc}"))
    if full.pixels:
        write_json(output / "metrics_summary.json", dict(protocol="raw metric, no GT alignment; positive finite GT/pred overlap",
            aggregation="global pixel-weighted", units="metres", full_image=full.result(),
            transparent_objects=transparent.result(), per_image=per_image, skipped=skipped,
            note="Sampled scene1-10 diagnostic, not an official held-out benchmark; no GT filling/filtering"))
    return skipped


def run(args, predictor_factory=DA3Predictor):
    # Set before importing torch/DA3/HF. A local directory is mandatory, so the
    # shared loader's online model_info/revision branch cannot be reached.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.depth_max <= 0 or args.error_max <= 0:
        raise ValueError("Colour limits must be positive")
    root = discover_root(args.data_root, args.scene_start, args.scene_end)
    protect_dataset(args.data_root, args.output_dir)
    protect_dataset(root, args.output_dir)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir() or not all((checkpoint / n).is_file() for n in ("config.json", "model.safetensors")):
        raise FileNotFoundError("Local --checkpoint must contain config.json and model.safetensors; downloads are disabled")
    calibration = load_calibration(root, args.camera, args.intrinsics, args.intrinsics_shape)
    selection = dict(root=str(root), scene_start=args.scene_start, scene_end=args.scene_end,
        camera=args.camera, num_samples=args.num_samples, seed=args.seed, sampling=args.sampling)
    config = dict(dataset="transcg", protocol_version=1, selection=selection,
        checkpoint=str(checkpoint), checkpoint_hashes={n: digest_file(checkpoint / n)
        for n in ("config.json", "model.safetensors")}, process_res=args.process_res,
        process_res_method="upper_bound_resize", device=args.device, precision=args.precision,
        calibration=calibration, single_image=True, gt_input=False, offline=True,
        canonical_units="official focal-normalized network output; not metres",
        metric_units="metres" if calibration else None)
    output = args.output_dir
    manifest_path, metadata_path = output / "sample_manifest.json", output / "run_metadata.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Existing TransCG run: use --resume or a fresh --output-dir; no outputs overwritten")
        manifest = read_json(manifest_path)
        if manifest.get("dataset") != "transcg" or manifest["selection"] != selection:
            raise ValueError("Saved sampling configuration differs; choose a fresh --output-dir")
        if metadata_path.exists() and read_json(metadata_path)["run_config"] != config:
            raise ValueError("Saved run configuration differs; choose a fresh --output-dir")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is nonempty and is not this TransCG run; refusing overwrite")
        _, candidates, scan = scan_rgb(root, args.scene_start, args.scene_end, args.camera)
        scenes = [f"scene{i}" for i in range(args.scene_start, args.scene_end + 1)]
        selected = sample_rgb(candidates, args.num_samples, args.seed, args.sampling, scenes)
        manifest = dict(dataset="transcg", schema_version=1, selection=selection, scan=scan,
            candidate_order="scene numeric order, then lexicographic relative RGB path",
            algorithm="Python random.Random(seed).sample, without replacement", samples=selected)
        write_json(manifest_path, manifest)
    print(f"Readable RGB candidates: {manifest['scan']['total_rgb']}; per scene: {manifest['scan']['per_scene']}")
    print(f"Seed: {args.seed}; sampling: {args.sampling}")
    for sample in manifest["samples"]:
        print(sample["path"], flush=True)
    if not calibration:
        print("WARNING: No confirmed camera intrinsics/pixel grid. Canonical-depth inference is allowed; scale is UNCALIBRATED, not metric. Trustworthy Raw Metric Evaluation is disabled.", flush=True)
    failures, records = [], []
    write_json(metadata_path, dict(run_config=config, model=None, status="initializing"))
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        import torch
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        predictor = predictor_factory(checkpoint=checkpoint, device=args.device, precision=args.precision)
        # Shared Booster metadata describes its metric output; override semantics
        # explicitly for this canonical-first workflow.
        model_metadata = dict(predictor.metadata, depth_semantics="canonical network output",
                              metric_output_enabled=calibration is not None)
        write_json(metadata_path, dict(run_config=config, model=model_metadata, status="running"))
        for sample in manifest["samples"]:
            folder = prediction_dir(output, sample)
            try:
                rgb = read_rgb(sample["path"])
                if list(rgb.shape[:2]) != sample["input_shape"] or digest_file(sample["path"]) != sample["input_sha256"]:
                    raise ValueError("Selected RGB changed since manifest creation; never replace selections")
                if args.resume and (folder / "prediction.json").exists():
                    saved = validate_saved(folder, sample, config)
                    records.append(dict(sample_id=sample["sample_id"], status="resumed", seconds=saved["seconds"],
                                        peak_cuda_allocated_bytes=saved["peak_cuda_allocated_bytes"]))
                    print(f"RESUMED {sample['sample_id']}", flush=True)
                    continue
                canonical, timing = predictor.predict_canonical(rgb, args.process_res, "upper_bound_resize")
                metric, conversion = calibrated_prediction(canonical, rgb.shape[:2], calibration)
                write_npy(folder / "canonical_depth.npy", canonical)
                names = ["canonical_depth.npy"]
                if metric is not None:
                    write_npy(folder / "metric_depth.npy", metric)
                    names.append("metric_depth.npy")
                _, _, annotations = read_annotations(sample, rgb.shape[:2])
                write_json(folder / "prediction.json", dict(**sample, run_config=config, status="success", error=None,
                    **timing, metric_units="metres" if metric is not None else None,
                    canonical_units=config["canonical_units"], metric_conversion=conversion,
                    calibration_source=calibration, annotations=annotations,
                    output_sha256={n: digest_file(folder / n) for n in names}))
                (folder / "failure.json").unlink(missing_ok=True)
                records.append(dict(sample_id=sample["sample_id"], status="success", **timing))
                print(f"OK {sample['sample_id']} {timing['seconds']:.3f}s", flush=True)
            except Exception as exc:
                failure = dict(sample_id=sample["sample_id"], path=sample["path"], stage="inference",
                               status="failed", error=f"{type(exc).__name__}: {exc}")
                failures.append(failure)
                records.append(failure)
                write_json(folder / "failure.json", failure)
                print(f"FAILED {sample['sample_id']}: {exc}", flush=True)
            finally:
                write_json(output / "inference_failures.json", failures)
                write_json(output / "inference_summary.json", dict(selected=len(manifest["samples"]),
                    records=records, failed=len(failures), pending=len(manifest["samples"]) - len(records)))
    except Exception as exc:
        failures.append(dict(stage="model_load", status="failed", error=f"{type(exc).__name__}: {exc}"))
        print(f"MODEL LOAD FAILED: {exc}", flush=True)
    finally:
        write_json(output / "inference_failures.json", failures)
        write_json(output / "inference_summary.json", dict(selected=len(manifest["samples"]), records=records,
            failed=len(failures), pending=len(manifest["samples"]) - len(records)))
    skipped = evaluate_saved(output, manifest["samples"], records, calibration is not None)
    figure_failures = regenerate_figures(output, args.depth_max, args.error_max)
    metadata = read_json(metadata_path)
    metadata.update(status="completed_with_failures" if failures or figure_failures else "completed",
                    evaluation_skipped=skipped, visualization_failures=figure_failures)
    write_json(metadata_path, metadata)
    return 1 if failures or figure_failures else 0


if __name__ == "__main__":
    args = parser().parse_args()
    if args.print_config:
        print(json.dumps(vars(args), indent=2, default=str))
    else:
        raise SystemExit(run(args))
