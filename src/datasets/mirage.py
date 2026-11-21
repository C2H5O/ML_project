"""Read a single Mirage-18k scene from a partial download."""

from pathlib import Path
import re
import warnings

import numpy as np
from PIL import Image


DEFAULT_SCENE = "2026-02-01_11-14-35"
FRAME_NAME = re.compile(r"[0-9]+_[0-9]+\.png")


def discover(data_root, scene):
    if not scene or Path(scene).name != scene or scene == "..":
        raise ValueError("--scene must be a single scene folder name")
    folder = Path(data_root) / scene / "rs_images"
    frames = []
    for path in sorted(folder.glob("*.png"), key=lambda path: path.name):
        if not path.is_file() or not FRAME_NAME.fullmatch(path.name):
            continue
        try:
            with Image.open(path) as image:
                if image.format == "PNG" and image.mode == "RGB":
                    frames.append(path)
        except OSError:
            warnings.warn(f"Skipping unreadable RGB file: {path}")
    if not frames:
        raise FileNotFoundError(f"No Mirage RGB frames found in {folder}")
    return frames


def select_samples(frames, count):
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)
    return [frames[index] for index in indices]


def read_annotations(rgb_path, shape):
    """Match by frame filename on the RGB grid. Official GT PNGs store millimetres."""
    scene = rgb_path.parent.parent

    def read(folder, modes):
        path = scene / folder / rgb_path.name
        if not path.is_file():
            return None
        try:
            with Image.open(path) as image:
                if (image.format == "PNG" and image.mode in modes
                        and image.size == (shape[1], shape[0])):
                    return np.array(image)
        except OSError:
            pass
        warnings.warn(f"Skipping unreadable annotation or RGB grid mismatch: {path}")
        return None

    # Pillow versions may expose 16-bit grayscale PNGs as I or I;16.
    gt = read("rs_corresponding_gt", {"I", "I;16"})
    mask = read("binary_masks", {"1", "L"})
    if gt is not None:
        gt = gt.astype(np.float32) / 1000.0 if (gt > 0).any() else None
    return gt, mask > 0 if mask is not None else None
