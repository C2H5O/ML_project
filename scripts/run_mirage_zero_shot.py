"""Run offline DA3METRIC-LARGE inference and visualize one Mirage-18k scene."""

import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image

import _bootstrap
from src.common import positive_int, protect_dataset, read_json, write_csv, write_npy
from src.datasets.mirage import DEFAULT_SCENE, discover, read_annotations, select_samples
from src.datasets.transcg import read_rgb
from src.evaluation.metrics import MetricAccumulator, valid_pixels
from src.inference.da3_zero_shot import (
    DA3Predictor, MODEL_ID, metric_depth_from_canonical, resize_prediction,
)
from src.visualization.transcg import colour_depth, labelled


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True, help="Path to the Mirage18k dataset root")
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--checkpoint", type=Path, default=_bootstrap.ROOT / "checkpoints/DA3METRIC-LARGE",
                   help="Local DA3METRIC-LARGE checkpoint directory")
    p.add_argument("--max-samples", type=positive_int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    p.add_argument("--output-dir", type=Path, default=_bootstrap.ROOT / "results/mirage_zero_shot")
    p.add_argument("--intrinsics", type=Path,
                   help="Optional JSON with calibrated K, width and height for this scene's RGB grid")
    return p


def grid(images, columns):
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    gap = 8
    canvas = Image.new("RGB", (columns * (width + gap) - gap, rows * (height + gap) - gap), "#e1e5ea")
    for index, image in enumerate(images):
        canvas.paste(image, (index % columns * (width + gap), index // columns * (height + gap)))
    return canvas


def depth_range(*arrays):
    values = np.concatenate([depth[np.isfinite(depth) & (depth > 0)] for depth in arrays])
    return tuple(np.percentile(values, [2, 98])) if values.size else (0.0, 1.0)


def render_sample(folder, frame, rgb, depth, gt, mask, metric):
    """Return an RGB/prediction overview tile and optional raw metric scores."""
    pred = resize_prediction(depth, rgb.shape[:2])
    limits = depth_range(pred, gt) if metric and gt is not None else depth_range(pred)
    pred_title = (f"DA3 prediction (m)\nDisplay {limits[0]:.2f}–{limits[1]:.2f} m" if metric
                  else "DA3 canonical depth\nPer-image normalization (p2–p98)")
    pred_colour = colour_depth(pred, *limits)
    rgb_image = Image.fromarray(rgb)
    rgb_title = f"RGB | {frame.name}"
    pred_panel = labelled(pred_colour, pred_title)
    pred_panel.save(folder / "prediction.png")

    panels = [labelled(rgb_image, rgb_title)]
    if gt is not None:
        gt_limits = limits if metric else depth_range(gt)
        gt_title = f"GT depth (m)\nDisplay {gt_limits[0]:.2f}–{gt_limits[1]:.2f} m"
        gt_panel = labelled(colour_depth(gt, *gt_limits), gt_title)
        gt_panel.save(folder / "gt_depth.png")
        panels.append(gt_panel)
    panels.append(pred_panel)
    if mask is not None:
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / "glass_mask.png")

    scores = None
    if metric and gt is not None:
        valid = valid_pixels(pred, gt)
        if valid.any():
            error = np.where(valid, np.abs(pred - gt), np.nan)
            error_max = float(np.percentile(error[valid], 98))
            error_panel = labelled(colour_depth(error, 0, error_max, cmap="magma"),
                                   f"Absolute error (m)\nDisplay 0–{error_max:.2f} m")
            error_panel.save(folder / "absolute_error.png")
            panels.append(error_panel)
            scores = MetricAccumulator().update(pred, gt).result()
    grid(panels, len(panels)).save(folder / "comparison.png")
    tile = grid([labelled(rgb_image, rgb_title, width=352),
                 labelled(pred_colour, pred_title, width=352)], 2)
    return tile, scores


def run(args):
    protect_dataset(args.data_root, args.output_dir)
    frames = discover(args.data_root, args.scene)
    selected = select_samples(frames, args.max_samples)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Choose a new or empty --output-dir to preserve existing results")
    if not all((args.checkpoint / name).is_file() for name in ("config.json", "model.safetensors")):
        raise FileNotFoundError(f"Local DA3METRIC-LARGE checkpoint missing in {args.checkpoint}")
    calibration = read_json(args.intrinsics) if args.intrinsics else None
    print(f"{args.scene}: selected {len(selected)} of {len(frames)} RGB frames", flush=True)
    if calibration is None:
        print("No calibrated RGB intrinsics supplied. Saving canonical DA3 depth.", flush=True)
    predictor = DA3Predictor(model_id=MODEL_ID, checkpoint=args.checkpoint.resolve(),
                             device=args.device, precision=args.precision, require_gpu=None)

    tiles, metrics = [], []
    for index, frame in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {frame.name}", flush=True)
        rgb = read_rgb(frame)
        if calibration is not None and (calibration["height"], calibration["width"]) != rgb.shape[:2]:
            raise ValueError(f"Intrinsics width/height do not match {frame.name}")
        canonical, _ = predictor.predict_canonical(rgb, process_res=560)
        metric = None
        if calibration is not None:
            metric, _ = metric_depth_from_canonical(canonical, calibration["K"], rgb.shape[:2])

        folder = args.output_dir / frame.stem
        folder.mkdir(parents=True)
        copyfile(frame, folder / "rgb.png")
        write_npy(folder / "prediction_canonical.npy", canonical)
        if metric is not None:
            write_npy(folder / "prediction_metres.npy", metric)
        gt, mask = read_annotations(frame, rgb.shape[:2])
        depth = metric if metric is not None else canonical
        tile, scores = render_sample(folder, frame, rgb, depth, gt, mask, metric is not None)
        tiles.append(tile)
        if scores is not None:
            metrics.append(dict(frame=frame.name, **scores))

    grid(tiles, min(2, len(tiles))).save(args.output_dir / "overview.png")
    if metrics:
        write_csv(args.output_dir / "metrics.csv", metrics, metrics[0].keys())
    print(f"Results saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    cli = parser()
    try:
        run(cli.parse_args())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        cli.exit(1, f"Mirage inference: {error}\n")
