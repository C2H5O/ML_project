"""Read schema-v1 outputs of booster_preprocess_code without reprocessing.

RGB stays uint8 HWC for the official DA3 InputProcessor. Depth and masks
stay at the saved processed size. No normalisation, crop, resize or conversion.
"""
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from src.common import read_json

MATERIAL_CLASSES = {
    0: "opaque_or_diffuse",
    1: "strongly_distorted_reflection_or_transmission",
    2: "partially_distorted_reflection_or_transmission",
    3: "regular_specular_or_transparent",
}
MATERIAL_SOURCE = "https://arxiv.org/html/2301.08245v3#S3"


def contained(root, relative):
    relative = str(relative).replace("\\", "/")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"Path must remain below dataset directory: {relative}")
    return path


class BoosterDataset:
    def __init__(self, data_root, split="train", scenes=None, subset=None):
        self.data_root = Path(data_root)
        if len(PurePosixPath(split).parts) != 1 or split in {".", ".."}:
            raise ValueError("split must be a single directory name")
        self.split = split
        self.root = contained(self.data_root, split)
        self.index = read_json(self.root / "index.json")
        if self.index.get("schema_version") != 1 or self.index.get("dataset") != "booster":
            raise ValueError("Expected Booster schema_version=1 index")
        if self.index.get("split") != split:
            raise ValueError("Requested split disagrees with index")
        entries = self.index["samples"]
        if self.index.get("num_samples") != len(entries):
            raise ValueError("index num_samples does not match entries")
        ids = [entry["sample_id"] for entry in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample IDs")
        for sid in ids:
            if len(PurePosixPath(sid).parts) != 3:
                raise ValueError(f"Expected subset/scene/camera sample ID: {sid}")
        self.samples = [entry for entry in entries
                        if (not scenes or PurePosixPath(entry["sample_id"]).parts[1] in scenes)
                        and (not subset or PurePosixPath(entry["sample_id"]).parts[0] == subset)]
        if not self.samples:
            raise ValueError("No samples for the requested split/scene/subset selection")

    def __len__(self):
        return len(self.samples)

    def sample_id(self, index):
        return self.samples[index]["sample_id"]

    def metadata(self, index):
        entry = self.samples[index]
        meta = read_json(contained(self.root, entry["meta"]))
        if meta.get("sample_id") != entry["sample_id"] or meta.get("split") != self.split:
            raise ValueError("Index and metadata identity mismatch")
        if meta.get("depth_unit") != "m":
            raise ValueError("Unconfirmed depth unit: expected explicit depth_unit='m'")
        return meta

    def paths(self, index, meta=None):
        meta = self.metadata(index) if meta is None else meta
        directory = contained(self.root, self.samples[index]["dir"])
        return {key: contained(directory, value) for key, value in meta["files"].items()}

    def rgb_input(self, index):
        """Inference only reads RGB and metadata; GT/masks never enter the model."""
        meta = self.metadata(index)
        paths = self.paths(index, meta)
        with Image.open(paths["rgb"]) as image:
            rgb = np.array(image.convert("RGB"))
        if list(rgb.shape[:2]) != meta["processed_size"]:
            raise ValueError("RGB shape disagrees with metadata")
        self.calibrated_K(meta)
        return rgb, meta, paths

    @staticmethod
    def calibrated_K(meta):
        """Require the supplied saved K to match its recorded resize/crop."""
        K = np.asarray(meta["K"], dtype=np.float64)
        original = np.asarray(meta["K_original"], dtype=np.float64)
        t = meta["transform"]
        oh, ow = meta["original_size"]
        rh, rw = t["resized_size"]
        h, w = meta["processed_size"]
        if (min(oh, ow, rh, rw, h, w) <= 0 or t["crop_x"] < 0 or t["crop_y"] < 0 or
                t["crop_x"] + w > rw or t["crop_y"] + h > rh):
            raise ValueError("Invalid saved resize/crop calibration geometry")
        matrix = np.array([[rw/ow, 0, -t["crop_x"]], [0, rh/oh, -t["crop_y"]], [0, 0, 1]])
        if (K.shape != (3, 3) or original.shape != (3, 3) or
                not np.all(np.isfinite(K)) or not np.all(np.isfinite(original)) or
                not np.allclose(K[2], [0, 0, 1]) or K[0, 0] <= 0 or K[1, 1] <= 0 or
                not np.allclose([t["scale_x"], t["scale_y"]], [rw/ow, rh/oh]) or
                not np.allclose(K, matrix @ original, atol=1e-5)):
            raise ValueError("Saved K is inconsistent with original calibration and resize/crop")
        return K

    def __getitem__(self, index):
        rgb, meta, paths = self.rgb_input(index)
        depth = np.load(paths["depth_gt"], allow_pickle=False)
        if depth.ndim != 2 or depth.dtype not in (np.dtype("float32"), np.dtype("float16")):
            raise ValueError("Expected HxW floating point depth array")
        valid = self._mask(paths["valid_mask"], depth.shape, binary=True)
        consistency = self._mask(paths["mask"], depth.shape, binary=True)
        material = None
        if meta.get("has_material_mask"):
            material = self._mask(paths["material_mask"], depth.shape, binary=False)
            if not set(np.unique(material)).issubset(MATERIAL_CLASSES):
                raise ValueError("Unknown material IDs; evaluation blocked")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("RGB and GT spatial sizes disagree")
        K = np.asarray(meta["K"], dtype=np.float64)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("Invalid saved intrinsics")
        return dict(rgb=rgb, depth_gt=depth,
                    valid_mask=valid & np.isfinite(depth) & (depth > 0),
                    saved_valid_mask=valid, consistency_mask=consistency,
                    material_mask=material, has_material_mask=material is not None,
                    updated_K=K, original_K=np.asarray(meta["K_original"]),
                    original_size=tuple(meta["original_size"]), processed_size=depth.shape,
                    scene_id=PurePosixPath(meta["sample_id"]).parts[1],
                    sample_id=meta["sample_id"], meta=meta)

    @staticmethod
    def _mask(path, shape, binary):
        with Image.open(path) as image:
            array = np.array(image)
        # Existing preprocessing cvtColor writes single-channel masks as RGB.
        # Collapse only identical channels; reject coloured/unconfirmed encodings.
        if array.ndim == 3 and array.shape[2] == 3:
            if not (np.array_equal(array[..., 0], array[..., 1]) and
                    np.array_equal(array[..., 0], array[..., 2])):
                raise ValueError(f"Unconfirmed colour mask encoding: {path.name}")
            array = array[..., 0]
        if array.shape != shape or array.ndim != 2:
            raise ValueError(f"Mask/GT size mismatch: {path.name}")
        if binary:
            if not set(np.unique(array)).issubset({0, 255}):
                raise ValueError(f"Expected saved 0/255 mask: {path.name}")
            return array > 0
        return array
