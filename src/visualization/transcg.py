"""TransCG panels and complete manifest contact sheets; no GT scale fitting."""
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont

from src.datasets.transcg import read_rgb, read_annotations, prediction_dir
from src.inference.da3_zero_shot import resize_prediction
from src.common import read_json, write_json, digest_file


def colour_depth(depth, low, high, cmap="viridis"):
    import matplotlib
    valid = np.isfinite(depth) & (depth > 0 if cmap == "viridis" else depth >= 0)
    normalized = np.clip((np.where(valid, depth, low) - low) / max(high - low, 1e-12), 0, 1)
    rgb = (matplotlib.colormaps[cmap](normalized)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 150
    return Image.fromarray(rgb)


@lru_cache(maxsize=1)
def panel_font():
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 18)


def labelled(image, title, width=640):
    image = image.resize((width, max(1, round(image.height * width / image.width))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, image.height + 48), "white")
    canvas.paste(image, (0, 48))
    ImageDraw.Draw(canvas).text((8, 3), title, fill="black", font=panel_font())
    return canvas


def save_transcg_panel(sample, canonical, metric, path, depth_max=2.0, error_max=1.0, zoom_regions=False):
    rgb = read_rgb(sample["path"])
    shape = rgb.shape[:2]
    gt, mask, annotations = read_annotations(sample, shape)
    error = None
    panels = [labelled(Image.fromarray(rgb), f"{sample['sample_id']}\nRGB input ({sample['camera_name']})")]
    settings = dict(depth_colormap="viridis", error_colormap="magma", invalid_colour="grey",
                    gt_alignment=False, annotations=annotations)
    if gt is not None:
        panels.append(labelled(colour_depth(gt, 0, depth_max), f"Refined GT (m): 0 .. {depth_max:g}\nNot raw sensor depth; grey = invalid"))
    if metric is None:
        prediction = resize_prediction(canonical, shape)
        valid = prediction[np.isfinite(prediction) & (prediction > 0)]
        low, high = np.percentile(valid, [2, 98]) if valid.size else (0, 1)
        if high <= low:
            high = low + 1
        prediction_image = colour_depth(prediction, low, high)
        panels.append(labelled(prediction_image,
                      f"DA3 canonical: {low:.3g} .. {high:.3g}\nUNCALIBRATED, not metres; per-image P2/P98"))
        settings.update(prediction_units="canonical, uncalibrated", prediction_range=[float(low), float(high)])
        if gt is not None:
            blank = Image.new("RGB", (shape[1], shape[0]), "lightgrey")
            panels.append(labelled(blank, "Absolute error unavailable\nMissing calibrated prediction scale"))
    else:
        prediction = resize_prediction(metric, shape)
        prediction_image = colour_depth(prediction, 0, depth_max)
        panels.append(labelled(prediction_image,
                      f"DA3 metric (m): 0 .. {depth_max:g}\nShared GT range; values above max saturate"))
        settings.update(prediction_units="metres", prediction_range=[0, depth_max])
        if gt is not None:
            valid = np.isfinite(gt) & (gt > 0) & np.isfinite(prediction) & (prediction > 0)
            error = np.full(shape, np.nan, np.float32)
            error[valid] = np.abs(prediction[valid] - gt[valid])
            panels.append(labelled(colour_depth(error, 0, error_max, "magma"),
                          f"Absolute error (m): 0 .. {error_max:g}\nValid GT/prediction overlap only"))
    if mask is not None:
        overlay = rgb.copy()
        overlay[mask] = (overlay[mask].astype(np.float32) * .5 + np.array([255, 80, 0]) * .5).astype(np.uint8)
        panels.append(labelled(Image.fromarray(overlay), "Transparent object mask overlay\nOrange = nonzero mask, not validity mask"))
    result = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
    offset = 0
    for panel in panels:
        result.paste(panel, (offset, 0))
        offset += panel.width
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)
    folder = prediction_dir(path.parent.parent, sample)
    prediction_image.save(folder / "predicted_depth.png")
    if error is not None:
        colour_depth(error, 0, error_max, "magma").save(folder / "absolute_error.png")
    if zoom_regions and metric is not None and mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        pad = max(1, round(max(ys.max() - ys.min(), xs.max() - xs.min()) * .08))
        box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
               min(shape[1], int(xs.max()) + pad + 1), min(shape[0], int(ys.max()) + pad + 1))
        crops = [(Image.fromarray(rgb), "RGB | transparent mask bounding box"),
                 (prediction_image, f"DA3 metric (m): 0 .. {depth_max:g}")]
        if gt is not None:
            crops.insert(1, (colour_depth(gt, 0, depth_max), f"Refined GT (m): 0 .. {depth_max:g}"))
        if error is not None:
            crops.append((colour_depth(error, 0, error_max, "magma"), f"Absolute error (m): 0 .. {error_max:g}"))
        crop_panels = [labelled(im.crop(box), title) for im, title in crops]
        zoom = Image.new("RGB", (sum(p.width for p in crop_panels), max(p.height for p in crop_panels)), "white")
        for i, crop in enumerate(crop_panels):
            zoom.paste(crop, (i * 640, 0))
        zoom_path = path.with_name(path.stem + "_transparent_zoom.png")
        zoom.save(zoom_path)
        settings["transparent_zoom"] = dict(path=str(zoom_path), bbox_xyxy=list(box),
            selection="nonzero transparent mask bounding box with padding; no depth-based ROI selection")
    write_json(folder / "visualization.json", settings)
    return settings


def regenerate_figures(output, depth_max=2.0, error_max=1.0, zoom_regions=False):
    """CPU-only render from saved predictions; keep a slot for every selection."""
    output = Path(output)
    manifest = read_json(output / "sample_manifest.json")
    tiles, records, failures = [], [], []
    for index, sample in enumerate(manifest["samples"], 1):
        path = output / "figures" / f"sample_{index:03d}.png"
        try:
            folder = prediction_dir(output, sample)
            meta = read_json(folder / "prediction.json")
            if meta["status"] != "success" or meta["sample_id"] != sample["sample_id"]:
                raise ValueError("No successful prediction for this selected sample")
            if digest_file(sample["path"]) != meta["input_sha256"]:
                raise ValueError("RGB changed since inference")
            for name, checksum in meta["output_sha256"].items():
                if digest_file(folder / name) != checksum:
                    raise ValueError(f"Prediction checksum mismatch: {name}")
            canonical = np.load(folder / "canonical_depth.npy", allow_pickle=False)
            metric = np.load(folder / "metric_depth.npy", allow_pickle=False) if meta["metric_units"] == "metres" else None
            settings = save_transcg_panel(sample, canonical, metric, path, depth_max, error_max, zoom_regions)
            with Image.open(path) as panel:
                tile = ImageOps.contain(panel.convert("RGB"), (1000, 300))
            records.append(dict(sample_id=sample["sample_id"], figure=str(path), **settings))
        except Exception as exc:
            failure = dict(sample_id=sample["sample_id"], error=f"{type(exc).__name__}: {exc}")
            failures.append(failure)
            path.unlink(missing_ok=True)  # Never leave a stale successful panel.
            tile = Image.new("RGB", (1000, 220), "lightgrey")
            ImageDraw.Draw(tile).text((10, 10), f"sample_{index:03d}: {sample['sample_id']}\nFAILED / unavailable\n{str(exc)[:100]}", fill="black", font=panel_font())
        slot = Image.new("RGB", (1000, 320), "white")
        ImageDraw.Draw(slot).text((8, 0), f"sample_{index:03d} | {sample['sample_id']}", fill="black", font=panel_font())
        slot.paste(tile, ((1000 - tile.width) // 2, 20))
        tiles.append(slot)
    columns = 2
    sheet = Image.new("RGB", (columns * 1000, max(1, math.ceil(len(tiles) / columns)) * 320), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * 1000, (i // columns) * 320))
    sheet.save(output / "contact_sheet.png")
    write_json(output / "visualization_summary.json", dict(records=records, failures=failures,
               depth_max_m=depth_max, error_max_m=error_max, selected=len(tiles)))
    return failures
