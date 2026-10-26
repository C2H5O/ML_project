"""Read-only TransCG discovery, reproducible sampling and optional annotations.

Protocol sources (no dependency on the TransCG training package):
https://graspnet.net/transcg
https://github.com/Galaxies99/TransCG/blob/main/utils/data_preparation.py
https://github.com/Galaxies99/TransCG/blob/main/datasets/transcg.py
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.common import digest_file
from src.inference.da3_zero_shot import metric_depth_from_canonical

CAMERAS = {1: "D435", 2: "L515"}
INTRINSICS = {1: "1-camIntrinsics-D435.npy", 2: "2-camIntrinsics-L515.npy"}


def discover_root(root, scene_start=1, scene_end=10):
    root = Path(root).resolve()
    if not 1 <= scene_start <= scene_end <= 10:
        raise ValueError("This protocol is limited to scene1 through scene10")
    layouts = [root, root / "data", root / "transcg" / "data", root / "transcg"]
    matches = [p for p in layouts if any((p / f"scene{i}").is_dir()
               for i in range(scene_start, scene_end + 1))]
    if len(matches) != 1:
        raise ValueError(f"Expected one unambiguous TransCG scene root; found {matches}")
    return matches[0]


def read_rgb(path):
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB PNG, got {image.mode}: {path}")
        return np.array(image, dtype=np.uint8)


def describe_sample(path, root, camera):
    path, root = Path(path).resolve(), Path(root).resolve()
    relative = path.relative_to(root)
    if (camera not in CAMERAS or path.name != f"rgb{camera}.png" or
            len(relative.parts) < 3 or not relative.parts[0].startswith("scene")):
        raise ValueError(f"Not a TransCG RGB frame: {relative}")
    scene = relative.parts[0]
    if scene not in {f"scene{i}" for i in range(1, 11)}:
        raise ValueError(f"Out of protocol scene: {scene}")
    frame = Path(*relative.parts[1:-1]).as_posix()
    return dict(path=str(path), relative_path=relative.as_posix(), scene_id=scene,
                frame_id=frame, camera_id=camera, camera_name=CAMERAS[camera],
                sample_id=f"{scene}/{frame}/camera{camera}")


def scan_rgb(root, scene_start=1, scene_end=10, camera=1):
    if camera not in CAMERAS:
        raise ValueError("camera must be 1 (D435) or 2 (L515)")
    root = discover_root(root, scene_start, scene_end)
    candidates, rejected, counts = [], [], {}
    for i in range(scene_start, scene_end + 1):
        scene = f"scene{i}"
        counts[scene] = 0
        # Exact basename excludes depth, GT, masks and the other camera.
        for path in sorted((root / scene).rglob(f"rgb{camera}.png")):
            try:
                sample = describe_sample(path, root, camera)
                rgb = read_rgb(path)
                sample.update(input_shape=list(rgb.shape[:2]), input_sha256=digest_file(path))
                if any(s["path"] == sample["path"] for s in candidates):
                    raise ValueError("Duplicate resolved image path")
                candidates.append(sample)
                counts[scene] += 1
            except Exception as exc:
                rejected.append(dict(path=str(path), error=f"{type(exc).__name__}: {exc}"))
    return root, candidates, dict(total_rgb=len(candidates), per_scene=counts,
                                  rejected_rgb=rejected)


def sample_rgb(candidates, count=20, seed=42, sampling="global", scenes=None):
    if count < 1 or count > len(candidates):
        raise ValueError(f"Cannot sample {count} unique images from {len(candidates)} candidates")
    if len({s["path"] for s in candidates}) != len(candidates):
        raise ValueError("Candidate list contains duplicate images")
    rng = random.Random(seed)
    if sampling == "global":
        return rng.sample(candidates, count)
    if sampling != "per_scene":
        raise ValueError("Unknown sampling mode")
    scenes = scenes or sorted({s["scene_id"] for s in candidates})
    if count % len(scenes):
        raise ValueError("per_scene requires num-samples divisible by number of requested scenes")
    selected = []
    for scene in scenes:
        pool = [s for s in candidates if s["scene_id"] == scene]
        n = count // len(scenes)
        if len(pool) < n:
            raise ValueError(f"{scene} has {len(pool)} images; needs {n}; no replacement allowed")
        selected.extend(rng.sample(pool, n))
    return selected


def prediction_dir(output, sample):
    # Stable collision-resistant path, never use an unchecked frame ID as a path.
    from src.common import sample_output
    return sample_output(output, sample["scene_id"], sample["sample_id"])


def validate_intrinsics(path, camera, image_shape=None):
    """Validate pixel K, never a pose/rotation. Grid correspondence is declared
    by the user, then checked against the actual RGB, not inferred from GT.
    """
    path = Path(path)
    if camera not in CAMERAS:
        raise ValueError("Unknown camera")
    name = path.name.lower()
    if (path.name in INTRINSICS.values() and path.name != INTRINSICS[camera] or
            ("l515" in name and camera != 2) or ("d435" in name and camera != 1)):
        raise ValueError("Intrinsics filename belongs to the other camera")
    if any(word in name for word in ("extrinsic", "pose", "t_camera", "rotation")):
        raise ValueError("Extrinsics/pose cannot be used as intrinsics")
    K = np.load(path, allow_pickle=False)
    if (K.dtype.kind not in "fiu" or K.shape != (3, 3) or
            not np.all(np.isfinite(K))):
        raise ValueError(f"Expected finite numeric 3x3 intrinsics, got {K.shape}, {K.dtype}")
    K = K.astype(np.float64)
    if (not np.allclose(K[2], [0, 0, 1], atol=1e-8) or
            not np.isclose(K[1, 0], 0, atol=1e-8) or
            K[0, 0] <= 0 or K[1, 1] <= 0 or
            np.allclose(K.T @ K, np.eye(3), atol=1e-6)):
        raise ValueError("Not a calibrated pinhole intrinsics matrix (possibly extrinsics/rotation)")
    if image_shape is not None:
        if (len(image_shape) != 2 or any(type(v) is not int or v <= 0 for v in image_shape)):
            raise ValueError("Intrinsics pixel grid must be positive integer HEIGHT WIDTH")
        h, w = image_shape
        if not (0 <= K[0, 2] < w and 0 <= K[1, 2] < h):
            raise ValueError("Principal point is outside the declared intrinsics pixel grid")
    return K


def calibration_candidates(root, camera):
    """Recursive discovery of official camera-specific basenames only."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted({p.resolve() for p in root.rglob(INTRINSICS[camera]) if p.is_file()})


def load_calibration(root, camera, intrinsics=None, intrinsics_shape=None, calibration_root=None):
    if intrinsics is not None:
        path = Path(intrinsics)
        # Validate known camera mismatch even when the explicit file is absent.
        if path.name in INTRINSICS.values() and path.name != INTRINSICS[camera]:
            raise ValueError("Intrinsics filename belongs to the other camera")
        if not path.is_file():
            raise FileNotFoundError(path)
    else:
        candidates = calibration_candidates(calibration_root or root, camera)
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous calibration files {candidates}; select --intrinsics explicitly")
        if not candidates:
            return None
        path = candidates[0]
    K = validate_intrinsics(path, camera, intrinsics_shape)
    if intrinsics_shape is None:
        if intrinsics is not None:
            raise ValueError("--intrinsics requires --intrinsics-shape HEIGHT WIDTH")
        return None  # A naked K does not establish its pixel grid.
    return dict(K=K.tolist(), image_shape=list(intrinsics_shape), source=str(path.resolve()),
                sha256=digest_file(path), camera_id=camera,
                stored_dtype=str(np.load(path, allow_pickle=False).dtype),
                provenance="user-confirmed corresponding camera intrinsics and pixel grid")


def calibrated_prediction(canonical, rgb_shape, calibration):
    if calibration is None:
        return None, None
    if list(rgb_shape) != calibration["image_shape"]:
        raise ValueError("RGB dimensions differ from confirmed intrinsics pixel grid; refusing metric conversion")
    return metric_depth_from_canonical(canonical, calibration["K"], rgb_shape)


def read_annotations(sample, shape):
    """Only refined *-gt is GT; sensor depth is never substituted or loaded.

    Official process_depth divides D435 by 1000 and L515 by 4000.
    No inpainting, depth normalization, GT alignment or training outlier filtering.
    """
    folder = Path(sample["path"]).parent
    camera = sample["camera_id"]
    gt_path = folder / f"depth{camera}-gt.png"
    mask_path = folder / f"depth{camera}-gt-mask.png"
    info = dict(gt_path=str(gt_path), gt_kind="refined ground truth, not raw sensor depth",
                gt_units="metres", stored_units_per_metre=1000 if camera == 1 else 4000,
                mask_definition="nonzero = transparent object; zero = non-transparent/background; NOT GT validity",
                mask_semantics_source="https://graspnet.net/transcg",
                depth_units_source="https://github.com/Galaxies99/TransCG/blob/main/utils/data_preparation.py",
                gt_available=False, mask_available=False, warnings=[])
    gt, mask = None, None
    for path, kind in [(gt_path, "gt"), (mask_path, "mask")]:
        if not path.is_file():
            info["warnings"].append(f"Missing {kind}: {path}")
            continue
        try:
            with Image.open(path) as image:
                # Pillow versions may expose 16-bit PNG as int32; check encoded mode too.
                mode = image.mode
                array = np.array(image)
            if array.shape != tuple(shape):
                raise ValueError(f"{kind} shape {array.shape} differs from RGB {shape}; no GT resizing")
            if kind == "gt":
                if mode not in {"I;16", "I;16L", "I;16B", "I"} or array.min() < 0 or array.max() > 65535:
                    raise ValueError("Expected unsigned 16-bit GT PNG encoding")
                gt = array.astype(np.float32) / info["stored_units_per_metre"]
                info.update(gt_available=True, gt_dtype=str(array.dtype), gt_sha256=digest_file(path),
                            valid_gt_pixels=int((array > 0).sum()))
            else:
                if not set(np.unique(array)).issubset({0, 1, 255}):
                    raise ValueError("Unrecognized transparent object mask values")
                mask = array != 0
                info.update(mask_available=True, mask_sha256=digest_file(path), mask_dtype=str(array.dtype),
                            mask_values=[int(v) for v in np.unique(array)], transparent_pixels=int(mask.sum()))
        except Exception as exc:
            info["warnings"].append(f"{kind}: {type(exc).__name__}: {exc}")
    return gt, mask, info
