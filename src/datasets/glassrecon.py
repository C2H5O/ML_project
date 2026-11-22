"""Read-only loader for the published GlassRecon evaluation release."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image

from src.common import read_json
from src.evaluation.metrics import MetricAccumulator

SPLIT_MANIFEST = Path(__file__).resolve().parents[2] / "configs/glassrecon_splits.json"
DEPTH_SCALE = 0.00025


@dataclass(frozen=True)
class Sample:
    sample_id: str
    building: str
    difficulty: str
    paths: dict[str, Path]


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray
    gt_raw: np.ndarray
    glass: np.ndarray
    camera: np.ndarray
    depth_scale: float

    @property
    def gt_m(self):
        return self.gt_raw.astype(np.float32) * self.depth_scale


def discover_root(path):
    root = Path(path).resolve()
    for candidate in (root, root / "GlassRecon"):
        if (candidate / "images").is_dir():
            return candidate
    raise FileNotFoundError(f"GlassRecon images/ not found below {root}; use the extracted dataset")


def load_splits(root, manifest):
    hard_ids = set(manifest["difficulty"]["hard_ids"])
    splits = {}
    seen_ids, seen_buildings = set(), set()
    for split, buildings in manifest["splits"].items():
        samples = []
        for building, ids in buildings.items():
            if building in seen_buildings:
                raise ValueError(f"GlassRecon building appears in multiple splits: {building}")
            seen_buildings.add(building)
            for sample_id in ids:
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate GlassRecon sample in split manifest: {sample_id}")
                seen_ids.add(sample_id)
                paths = {
                    role: root / template.format(id=sample_id)
                    for role, template in manifest["source_paths"].items()
                }
                for path in paths.values():
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing GlassRecon file for {sample_id}: {path}")
                difficulty = "hard" if sample_id in hard_ids else "easy"
                samples.append(Sample(sample_id, building, difficulty, paths))
        samples.sort(key=lambda sample: sample.sample_id)
        random.Random(manifest["seed"]).shuffle(samples)
        splits[split] = samples

    images = {p.name for p in (root / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    expected = {Path(manifest["source_paths"]["rgb"].format(id=i)).name for i in seen_ids}
    if images != expected:
        raise ValueError(f"GlassRecon RGB files differ from the fixed split: {sorted(images ^ expected)[:5]}")
    if not hard_ids <= seen_ids:
        raise ValueError("GlassRecon difficulty IDs must appear in the split manifest")
    return splits


def read_frame(sample):
    with Image.open(sample.paths["rgb"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(sample.paths["mask"]) as image:
        mask = np.asarray(image)
    gt = np.load(sample.paths["gt"], allow_pickle=False)
    metadata = read_json(sample.paths["intrinsics"])
    shape = rgb.shape[:2]
    expected_shape = (metadata["height"], metadata["width"])
    if not shape == mask.shape == gt.shape == expected_shape:
        raise ValueError(f"{sample.sample_id}: RGB, mask, GT and calibration grids differ")
    if gt.dtype != np.uint16 or ((mask != 0) & (mask != 255)).any():
        raise ValueError(f"{sample.sample_id}: expected uint16 evaluation GT and a binary 0/255 mask")
    scale = metadata["depth_scale"]
    if scale != DEPTH_SCALE:
        raise ValueError(f"{sample.sample_id}: unverified GT depth scale {scale}; expected {DEPTH_SCALE} m/unit")
    camera = np.array([
        [metadata["fx"], 0, metadata["ppx"]],
        [0, metadata["fy"], metadata["ppy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    if not np.isfinite(camera).all() or min(camera[0, 0], camera[1, 1]) <= 0:
        raise ValueError(f"{sample.sample_id}: invalid calibrated pixel intrinsics")
    return Frame(rgb, gt, mask == 255, camera, scale)


def inspect_release(splits):
    """Decode the release on CPU and verify the published difficulty assignments."""
    shapes, gt_dtypes, difficulty = Counter(), Counter(), Counter()
    gt_pixels, positive_gt_pixels = 0, 0
    gt_ranges = []
    for samples in splits.values():
        for sample in samples:
            frame = read_frame(sample)
            gt = frame.gt_m
            with Image.open(sample.paths["sensor"]) as image:
                sensor = np.asarray(image, dtype=np.float32) * frame.depth_scale
            if sensor.shape != gt.shape:
                raise ValueError(f"{sample.sample_id}: sensor and GT grids differ")
            # The published evaluator uses finite depths above 1e-6 m in both maps.
            valid = np.isfinite(sensor) & (sensor > 1e-6) & (gt > 1e-6)
            if not valid.any():
                raise ValueError(f"{sample.sample_id}: no valid sensor/GT overlap for difficulty")
            absrel = np.mean(np.abs(sensor[valid] - gt[valid]) / gt[valid])
            group = "hard" if absrel > 0.03 else "easy"
            if group != sample.difficulty:
                raise ValueError(f"{sample.sample_id}: sensor/GT AbsRel disagrees with difficulty manifest")
            difficulty[group] += 1
            shapes[str(gt.shape)] += 1
            gt_dtypes[str(frame.gt_raw.dtype)] += 1
            positive = gt[gt > 0]
            gt_pixels += gt.size
            positive_gt_pixels += positive.size
            gt_ranges.append((float(positive.min()), float(positive.max())))

    gt = read_frame(splits["test"][0]).gt_m
    return {
        "samples": {key: len(samples) for key, samples in splits.items()},
        "buildings": {key: len({sample.building for sample in samples}) for key, samples in splits.items()},
        "difficulty": dict(difficulty),
        "gt_shapes": dict(shapes),
        "gt_storage_dtypes": dict(gt_dtypes),
        "gt_scale_m_per_unit": DEPTH_SCALE,
        "positive_gt_range_m": [min(lo for lo, _ in gt_ranges), max(hi for _, hi in gt_ranges)],
        "positive_gt_pixels": positive_gt_pixels,
        "total_gt_pixels": gt_pixels,
        "metric_identity_check": MetricAccumulator().update(gt, gt).result(),
    }
