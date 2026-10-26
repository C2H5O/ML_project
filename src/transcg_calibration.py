"""CPU-only provenance checks for existing TransCG predictions. No torch import."""
from pathlib import Path
import re

import numpy as np

from src.common import read_json, digest_file
from src.datasets.transcg import (read_rgb, describe_sample, prediction_dir,
    calibration_candidates, validate_intrinsics, load_calibration, read_annotations)
from src.inference.da3_zero_shot import MODEL_ID, DA3_COMMIT


def validate_source_run(source, checkpoint=None):
    source = Path(source).resolve()
    manifest = read_json(source / "sample_manifest.json")
    metadata = read_json(source / "run_metadata.json")
    config, model = metadata["run_config"], metadata["model"]
    if (manifest.get("dataset") != "transcg" or config.get("dataset") != "transcg" or
            config.get("protocol_version") != 1 or config.get("selection") != manifest.get("selection") or
            config.get("single_image") is not True or config.get("gt_input") is not False or
            config.get("process_res_method") != "upper_bound_resize"):
        raise ValueError("Unsupported or inconsistent source TransCG inference protocol")
    if config.get("calibration") is not None or config.get("metric_units") is not None:
        raise ValueError("Source is already calibrated; refusing a second metric conversion")
    if not isinstance(model, dict) or any([
        model.get("model_id") != MODEL_ID, model.get("architecture") != "da3metric-large",
        model.get("da3_code_commit") != DA3_COMMIT, model.get("gt_alignment") is not False,
        model.get("checkpoint_kind") != "local_hf_directory",
        model.get("depth_semantics") != "canonical network output",
        model.get("metric_output_enabled") is not False,
    ]):
        raise ValueError("Missing/unverified DA3METRIC-LARGE source model/checkpoint provenance")
    if config.get("canonical_units") != "official focal-normalized network output; not metres":
        raise ValueError("Unknown source canonical units")
    hashes = config.get("checkpoint_hashes", {})
    if (set(hashes) != {"config.json", "model.safetensors"} or
            any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in hashes.values()) or
            model.get("checkpoint_hashes") != hashes):
        raise ValueError("Missing or inconsistent checkpoint SHA256 provenance")
    checkpoint_report = dict(recorded_path=config.get("checkpoint"), recorded_hashes=hashes,
                             local_files_verified=False)
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        for name, expected in hashes.items():
            if digest_file(checkpoint / name) != expected:
                raise ValueError(f"Local checkpoint checksum differs from saved source: {name}")
        if read_json(checkpoint / "config.json").get("model_name") != "da3metric-large":
            raise ValueError("Local checkpoint is not DA3METRIC-LARGE")
        checkpoint_report.update(local_files_verified=True, verified_path=str(checkpoint.resolve()),
                                 method="streaming SHA256 only; weights never deserialized")
    samples = manifest.get("samples", [])
    if (not samples or len(samples) != manifest["selection"]["num_samples"] or
            len({s["sample_id"] for s in samples}) != len(samples) or
            len({s["path"] for s in samples}) != len(samples)):
        raise ValueError("Invalid/duplicate saved sample selection")
    if manifest["selection"]["camera"] != 1 or any(s["camera_id"] != 1 for s in samples):
        raise ValueError("CPU calibration protocol accepts only D435 camera1 RGB1")
    return manifest, metadata, checkpoint_report


def validate_source_sample(source, sample, config):
    # Parse original manifest paths rather than re-discovering/re-sampling data.
    parsed = describe_sample(sample["path"], config["selection"]["root"], 1)
    for key in ("sample_id", "relative_path", "scene_id", "frame_id", "camera_id", "camera_name", "path"):
        if sample.get(key) != parsed[key]:
            raise ValueError(f"Manifest path identity mismatch: {key}")
    folder = prediction_dir(source, sample)
    # Never follow a prediction-directory symlink outside the source run.
    if Path(source).resolve() not in folder.resolve().parents:
        raise ValueError("Prediction path escapes source run")
    metadata_path = folder / "prediction.json"
    saved = read_json(metadata_path)
    if (saved.get("status") != "success" or saved.get("run_config") != config or
            any(saved.get(k) != v for k, v in sample.items())):
        raise ValueError("Sample prediction metadata differs from saved manifest/config")
    if (saved.get("metric_units") is not None or saved.get("metric_conversion") is not None or
            saved.get("prediction_is_metric") not in (None, 0) or
            saved.get("canonical_units") != config["canonical_units"]):
        raise ValueError("Already metric/unknown units; refusing repeated canonical conversion")
    if set(saved.get("output_sha256", {})) != {"canonical_depth.npy"}:
        raise ValueError("Source must contain an audited canonical-only prediction")
    canonical_path = folder / "canonical_depth.npy"
    if digest_file(canonical_path) != saved["output_sha256"]["canonical_depth.npy"]:
        raise ValueError("Source canonical prediction checksum mismatch")
    rgb = read_rgb(sample["path"])
    if list(rgb.shape[:2]) != sample["input_shape"] or digest_file(sample["path"]) != sample["input_sha256"]:
        raise ValueError("Source RGB differs from saved inference input")
    canonical = np.load(canonical_path, allow_pickle=False)
    if (canonical.dtype != np.float32 or canonical.ndim != 2 or
            list(canonical.shape) != saved["native_shape"] or
            not (np.isfinite(canonical) & (canonical > 0)).any()):
        raise ValueError("Invalid source canonical array dtype/shape/values")
    return canonical, saved, canonical_path, metadata_path


def inspect_calibration(calibration_root, camera=1, intrinsics=None, intrinsics_shape=None, input_dir=None):
    """Report evidence and uncertainties; finding K does not prove its grid."""
    report = dict(camera_id=camera, calibration_root=str(calibration_root),
                  calibration_root_exists=Path(calibration_root).is_dir(), declared_intrinsics_shape=intrinsics_shape, candidates=[],
                  rgb_samples=[], rgb_errors=[], ready_for_metric=False,
                  grid_confirmation="intrinsics_shape must be confirmed from calibration provenance; RGB dimensions alone are insufficient")
    paths = [Path(intrinsics)] if intrinsics else calibration_candidates(calibration_root, camera)
    if not paths:
        report["discovery_warning"] = "No official camera-specific intrinsics file found; no focal length is guessed"
    for path in paths:
        row = dict(path=str(path))
        try:
            # Inspect raw shape/dtype/values before validation, including rejected extrinsics.
            array = np.load(path, allow_pickle=False)
            row.update(shape=list(array.shape), dtype=str(array.dtype),
                       values=array.tolist() if array.size <= 16 and array.dtype.kind in "fiu" and np.isfinite(array).all() else None)
            K = validate_intrinsics(path, camera, intrinsics_shape)
            row.update(valid_K=True, K=K.tolist(), sha256=digest_file(path))
        except Exception as exc:
            row.update(valid_K=False, error=f"{type(exc).__name__}: {exc}")
        report["candidates"].append(row)
    if input_dir is not None:
        try:
            manifest = read_json(Path(input_dir) / "sample_manifest.json")
            for sample in manifest["samples"]:
                try:
                    rgb = read_rgb(sample["path"])
                    if sample["camera_id"] != camera or digest_file(sample["path"]) != sample["input_sha256"]:
                        raise ValueError("RGB hash or camera differs from manifest")
                    shape = list(rgb.shape[:2])
                    if shape != sample["input_shape"]:
                        raise ValueError("Actual RGB dimensions differ from saved dimensions")
                    _, _, annotations = read_annotations(sample, shape)
                    report["rgb_samples"].append(dict(sample_id=sample["sample_id"], path=sample["path"],
                        actual_shape=shape, matches_declared_grid=shape == intrinsics_shape, annotations=annotations))
                except Exception as exc:
                    report["rgb_errors"].append(dict(sample_id=sample.get("sample_id"), error=str(exc)))
        except Exception as exc:
            report["rgb_errors"].append(dict(error=f"Cannot inspect original manifest/RGB: {exc}"))
    report["actual_rgb_shapes"] = sorted({tuple(row["actual_shape"]) for row in report["rgb_samples"]})
    try:
        calibration = load_calibration(calibration_root, camera, intrinsics, intrinsics_shape)
        report["ready_for_metric"] = (calibration is not None and bool(report["rgb_samples"]) and
            not report["rgb_errors"] and all(s["matches_declared_grid"] for s in report["rgb_samples"]))
        report["confirmed_calibration"] = calibration
    except Exception as exc:
        report["calibration_error"] = str(exc)
    return report
