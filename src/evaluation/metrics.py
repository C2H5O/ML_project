"""Float64 sufficient statistics for global pixel-weighted depth metrics."""
from dataclasses import dataclass
import numpy as np


def valid_pixels(pred, gt, mask=None):
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("pred and gt must be matching HxW arrays")
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    if mask is not None:
        if np.shape(mask) != gt.shape:
            raise ValueError("mask shape mismatch")
        valid &= np.asarray(mask, dtype=bool)
    return valid


def align_scale(pred, gt, mask=None, min_scale=1e-6, max_scale=1e6):
    valid = valid_pixels(pred, gt, mask)
    if not valid.any():
        raise ValueError("No valid pixels for evaluation-time GT alignment")
    p, g = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    median = float(np.median(p[valid]))
    if median <= np.finfo(np.float64).tiny:
        raise ValueError("Zero/underflow prediction median")
    scale = float(np.median(g[valid]) / median)
    if not np.isfinite(scale) or not min_scale <= scale <= max_scale:
        raise ValueError(f"Abnormal GT alignment scale: {scale}")
    aligned = p * scale
    if not np.all(np.isfinite(aligned[valid])):
        raise ValueError("Alignment overflow")
    return aligned, scale


@dataclass
class MetricAccumulator:
    pixels: int = 0
    images: int = 0
    sum_absrel: float = 0.0
    sum_squared: float = 0.0
    correct: int = 0

    def update(self, pred, gt, mask=None):
        valid = valid_pixels(pred, gt, mask)
        n = int(valid.sum())
        if n:
            p = np.asarray(pred, dtype=np.float64)[valid]
            g = np.asarray(gt, dtype=np.float64)[valid]
            difference = p - g
            self.pixels += n
            self.images += 1
            self.sum_absrel += float(np.sum(np.abs(difference) / g))
            self.sum_squared += float(np.sum(difference ** 2))
            self.correct += int(np.sum(np.maximum(p / g, g / p) < 1.25))
        return self

    def result(self):
        return dict(status="available" if self.pixels else "unavailable",
                    valid_pixels=self.pixels, samples=self.images,
                    AbsRel=self.sum_absrel / self.pixels if self.pixels else None,
                    RMSE=float(np.sqrt(self.sum_squared / self.pixels)) if self.pixels else None,
                    delta1=self.correct / self.pixels if self.pixels else None)


def region_masks(material, shape):
    masks = {"full_image": np.ones(shape, dtype=bool)}
    # Explicit project grouping, not separate transparent vs mirror labels.
    masks.update({name: np.zeros(shape, bool) for name in
                  ["target_classes_2_3", "ordinary_class_0", "other_classes_0_1"]})
    masks.update({f"material_{i}": np.zeros(shape, bool) for i in range(4)})
    if material is not None:
        if material.shape != shape or not set(np.unique(material)).issubset({0, 1, 2, 3}):
            raise ValueError("Unconfirmed material mask")
        masks.update(target_classes_2_3=np.isin(material, [2, 3]),
                     ordinary_class_0=material == 0, other_classes_0_1=material <= 1)
        masks.update({f"material_{i}": material == i for i in range(4)})
    return masks
