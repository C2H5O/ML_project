"""Run frozen DA3METRIC-LARGE on the fixed GlassRecon test split."""

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import digest_file, positive_int, protect_dataset, read_json, write_csv, write_json, write_npy
from src.datasets.glassrecon import SPLIT_MANIFEST, discover_root, inspect_release, load_splits, read_frame
from src.evaluation.metrics import MetricAccumulator, valid_pixels
from src.inference.da3_zero_shot import DA3Predictor, resize_prediction
from src.visualization.transcg import colour_depth, labelled, panel_font

PROCESS_RES = 560
REGIONS = ("full_image", "glass", "non_glass")
METRICS = ("AbsRel", "RMSE", "delta1")
DEFAULT_OUTPUT = Path("results/glassrecon_zero_shot")


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def score_regions(pred, gt, glass):
    gt_valid = np.isfinite(gt) & (gt > 0)
    for name, region in zip(REGIONS, (np.ones_like(glass), glass, ~glass)):
        region_pixels = int(region.sum())
        gt_pixels = int((gt_valid & region).sum())
        result = MetricAccumulator().update(pred, gt, region).result()
        valid = result["valid_pixels"]
        yield {
            "region": name,
            "status": "evaluated" if valid else "no_valid_prediction" if gt_pixels else "no_valid_gt",
            "region_pixels": region_pixels,
            "gt_valid_pixels": gt_pixels,
            "valid_pixels": valid,
            "gt_coverage": fraction(gt_pixels, region_pixels),
            "valid_coverage": fraction(valid, region_pixels),
            **{metric: result[metric] for metric in METRICS},
        }


def summarize(rows):
    for difficulty in ("all", "easy", "hard"):
        for region in REGIONS:
            matching = [r for r in rows if r["region"] == region and
                        (difficulty == "all" or r["difficulty"] == difficulty)]
            evaluated = [r for r in matching if r["valid_pixels"]]
            region_pixels = sum(r["region_pixels"] for r in matching)
            gt_pixels = sum(r["gt_valid_pixels"] for r in matching)
            valid = sum(r["valid_pixels"] for r in matching)
            yield {
                "difficulty": difficulty,
                "region": region,
                "images_selected": len(matching),
                "images_with_valid_gt": sum(r["gt_valid_pixels"] > 0 for r in matching),
                "images_evaluated": len(evaluated),
                "region_pixels": region_pixels,
                "gt_valid_pixels": gt_pixels,
                "valid_pixels": valid,
                "gt_coverage": fraction(gt_pixels, region_pixels),
                "valid_coverage": fraction(valid, region_pixels),
                **{metric: mean(r[metric] for r in evaluated) if evaluated else None for metric in METRICS},
            }


def save_panel(sample, frame, gt, pred, path):
    valid = valid_pixels(pred, gt)
    error = np.full(gt.shape, np.nan, dtype=np.float32)
    error[valid] = np.abs(pred[valid] - gt[valid])
    gt_positive = gt[gt > 0]
    pred_positive = pred[np.isfinite(pred) & (pred > 0)]
    depth_max = max(
        float(np.percentile(arr, 98)) for arr in (gt_positive, pred_positive, np.array([0.01])) if arr.size
    )
    error_max = max(float(np.percentile(error[valid], 98)), 0.01) if valid.any() else 1.0
    overlay = frame.rgb.copy()
    overlay[frame.glass] = 0.45 * overlay[frame.glass] + 0.55 * np.array([255, 92, 12])
    displays = [
        (Image.fromarray(frame.rgb), "RGB"),
        (Image.fromarray(overlay), "Glass mask\nOrange = glass"),
        (colour_depth(gt, 0, depth_max), f"GT depth (m)\n0–{depth_max:.2f} · grey = invalid"),
        (colour_depth(pred, 0, depth_max), f"DA3 depth (m)\n0–{depth_max:.2f} · grey = invalid"),
        (colour_depth(error, 0, error_max, "magma"), f"Absolute error (m)\n0–{error_max:.2f} · valid overlap"),
    ]
    panels = [labelled(image, title, width=320) for image, title in displays]
    banner = 32
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height + banner), "white")
    title = f"{sample.sample_id}  |  {sample.difficulty.title()}  |  RGB-only metric depth"
    ImageDraw.Draw(canvas).text((8, 7), title, fill="black", font=panel_font())
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * panel.width, banner))
    canvas.save(path)


def save_overview(rows, figures, limit=16):
    glass_rows = [r for r in rows if r["region"] == "glass"]
    by_difficulty = {
        group: sorted((r for r in glass_rows if r["difficulty"] == group),
                      key=lambda r: (r["AbsRel"] is None, r["AbsRel"] or 0, r["sample_id"]))
        for group in ("easy", "hard")
    }
    easy_count = min(len(by_difficulty["easy"]), max(limit // 2, limit - len(by_difficulty["hard"])))
    counts = (easy_count, min(len(by_difficulty["hard"]), limit - easy_count))
    chosen = []
    for group, count in zip(by_difficulty.values(), counts):
        indices = np.linspace(0, len(group) - 1, count).round().astype(int)
        chosen.extend(group[index]["sample_id"] for index in indices)

    # Keep both columns large enough to read the per-image ranges.
    width = 1280
    thumbnails = []
    for sample_id in chosen:
        with Image.open(figures / f"{sample_id}.png") as panel:
            height = round(panel.height * width / panel.width)
            thumbnails.append(panel.resize((width, height), Image.Resampling.LANCZOS))
    stride = max(im.height for im in thumbnails) + 8
    columns = min(2, len(thumbnails))
    overview = Image.new("RGB", (columns * (width + 8), ((len(thumbnails) + columns - 1) // columns) * stride),
                         "#e8e8e8")
    for index, thumbnail in enumerate(thumbnails):
        overview.paste(thumbnail, ((index % columns) * (width + 8), (index // columns) * stride))
    overview.save(figures / "overview.png")
    return chosen


def run(args):
    root = discover_root(args.data_root)
    manifest = read_json(SPLIT_MANIFEST)
    splits = load_splits(root, manifest)
    if args.inspect_only:
        print(json.dumps(inspect_release(splits), indent=2))
        return

    samples = splits[args.split][:args.max_samples]
    output = args.output_dir.resolve()
    protect_dataset(root, output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output-dir; already exists: {output}")
    predictor = DA3Predictor(checkpoint=args.checkpoint, device=args.device, precision=args.precision,
                             require_gpu=None)
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"
    figures.mkdir()
    rows = []
    for index, sample in enumerate(samples, 1):
        frame = read_frame(sample)
        canonical, metric_native, details = predictor.predict(frame.rgb, frame.camera, process_res=PROCESS_RES)
        gt = frame.gt_m
        pred = resize_prediction(metric_native, gt.shape)
        destination = output / "predictions" / sample.sample_id
        write_npy(destination / "canonical.npy", canonical)
        write_npy(destination / "metric_m.npy", pred)
        identity = {
            "sample_id": sample.sample_id,
            "building": sample.building,
            "difficulty": sample.difficulty,
            "focal_multiplier": details["metric_conversion"]["multiplier"],
        }
        rows.extend({**identity, **row} for row in score_regions(pred, gt, frame.glass))
        save_panel(sample, frame, gt, pred, figures / f"{sample.sample_id}.png")
        print(f"[{index}/{len(samples)}] {sample.sample_id}", flush=True)

    summary = list(summarize(rows))
    chosen = save_overview(rows, figures)
    write_csv(output / "metrics_per_image.csv", rows, rows[0].keys())
    write_csv(output / "metrics_summary.csv", summary, summary[0].keys())
    write_json(output / "metrics_summary.json", {
        "model": predictor.metadata,
        "process_res": PROCESS_RES,
        "split": args.split,
        "split_manifest_sha256": digest_file(SPLIT_MANIFEST),
        "samples_in_split": len(splits[args.split]),
        "images_evaluated": summary[0]["images_evaluated"],
        "gt_depth_scale_m_per_unit": frame.depth_scale,
        "aggregation": "arithmetic mean of available per-image metrics by difficulty and region",
        "coverage": "GT-valid or jointly valid pixels divided by total region pixels",
        "overview_sample_ids": chosen,
        "metrics": summary,
    })
    print(f"Saved GlassRecon results to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Local DA3METRIC-LARGE Hugging Face directory")
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--max-samples", type=positive_int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inspect-only", action="store_true", help="Verify the full release on CPU")
    args = parser.parse_args()
    if not args.inspect_only and args.checkpoint is None:
        parser.error("--checkpoint is required for inference")
    run(args)


if __name__ == "__main__":
    main()
