"""Synthetic CPU postprocessing fixtures only; no real predictions/checkpoints."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from src.common import read_json, write_json, write_npy, digest_file
from src.datasets.transcg import (scan_rgb, prediction_dir, load_calibration,
    validate_intrinsics, calibrated_prediction)
from src.transcg_calibration import inspect_calibration, validate_source_run
from src.inference.da3_zero_shot import MODEL_ID, DA3_COMMIT, metric_depth_from_canonical


@pytest.fixture
def cpu_runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("calibrate_transcg_test", scripts / "calibrate_transcg_predictions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(scripts))


def snapshot(root):
    return {p.relative_to(root).as_posix(): digest_file(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def source_run(tmp_path):
    root, source, checkpoint = tmp_path / "rgb_data", tmp_path / "zero_shot", tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_json(checkpoint / "config.json", {"model_name": "da3metric-large"})
    (checkpoint / "model.safetensors").write_text("SYNTHETIC TEST CHECKPOINT, NEVER LOADED")
    hashes = {n: digest_file(checkpoint / n) for n in ("config.json", "model.safetensors")}
    for frame in range(2):
        folder = root / "scene1" / str(frame)
        folder.mkdir(parents=True)
        Image.fromarray(np.full((4, 6, 3), 60, np.uint8)).save(folder / "rgb1.png")
        gt = np.full((4, 6), 1000, np.uint16)
        gt[-1, -1] = 0  # Invalid GT must not enter metrics or error visualization.
        Image.fromarray(gt).save(folder / "depth1-gt.png")
        Image.fromarray(np.full((4, 6), 10000, np.uint16)).save(folder / "depth1.png")
        mask = np.zeros((4, 6), np.uint8)
        mask[:, :3] = 1
        Image.fromarray(mask).save(folder / "depth1-gt-mask.png")
    _, samples, scan = scan_rgb(root, 1, 1)
    selection = dict(root=str(root.resolve()), scene_start=1, scene_end=1, camera=1,
                     num_samples=2, seed=42, sampling="global")
    config = dict(dataset="transcg", protocol_version=1, selection=selection,
        checkpoint=str(checkpoint.resolve()), checkpoint_hashes=hashes,
        process_res=560, process_res_method="upper_bound_resize", device="cuda:0", precision="bf16",
        calibration=None, single_image=True, gt_input=False, offline=True,
        canonical_units="official focal-normalized network output; not metres", metric_units=None)
    model = dict(model_id=MODEL_ID, architecture="da3metric-large", da3_code_commit=DA3_COMMIT,
        gt_alignment=False, checkpoint_kind="local_hf_directory", checkpoint_hashes=hashes,
        depth_semantics="canonical network output", metric_output_enabled=False, test_fixture_only=True)
    write_json(source / "sample_manifest.json", dict(dataset="transcg", schema_version=1,
        selection=selection, samples=samples, scan=scan))
    write_json(source / "run_metadata.json", dict(run_config=config, model=model, status="completed"))
    for sample in samples:
        folder = prediction_dir(source, sample)
        array = np.array([[2, 4, 4], [2, 4, 4]], np.float32)
        write_npy(folder / "canonical_depth.npy", array)
        write_json(folder / "prediction.json", dict(**sample, status="success", run_config=config,
            canonical_units=config["canonical_units"], metric_units=None, metric_conversion=None,
            prediction_is_metric=None, native_shape=[2, 3], seconds=.123, peak_cuda_allocated_bytes=123,
            output_sha256={"canonical_depth.npy": digest_file(folder / "canonical_depth.npy")}))
    camera_root = tmp_path / "camera_info"
    path = camera_root / "nested" / "transcg" / "camera_intrinsics" / "1-camIntrinsics-D435.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.array([[300, 0, 3], [0, 300, 2], [0, 0, 1]], np.float32))
    return dict(root=root, source=source, checkpoint=checkpoint, camera_root=camera_root,
                K_path=path, samples=samples, config=config, output=tmp_path / "calibrated")


def arguments(cpu_runner, fixture):
    return cpu_runner.parser().parse_args(["--input-dir", str(fixture["source"]),
        "--output-dir", str(fixture["output"]), "--calibration-root", str(fixture["camera_root"]),
        "--intrinsics-shape", "4", "6", "--checkpoint", str(fixture["checkpoint"])])


def test_cpu_calibration_preserves_source_and_uses_raw_scale(source_run, cpu_runner, monkeypatch):
    f = source_run
    before = snapshot(f["source"])
    dataset_before, camera_before = snapshot(f["root"]), snapshot(f["camera_root"])
    # Any accidental model construction would fail this CPU-only integration test.
    from src.inference.da3_zero_shot import DA3Predictor
    def forbidden(*a, **kw):
        raise AssertionError("No model may be called")
    monkeypatch.setattr(DA3Predictor, "__init__", forbidden)
    assert cpu_runner.run(arguments(cpu_runner, f)) == 0
    assert snapshot(f["source"]) == before
    assert snapshot(f["root"]) == dataset_before
    assert snapshot(f["camera_root"]) == camera_before
    assert (f["output"] / "sample_manifest.json").read_bytes() == (f["source"] / "sample_manifest.json").read_bytes()
    calibration = load_calibration(f["root"], 1, intrinsics_shape=[4, 6], calibration_root=f["camera_root"])
    for sample in f["samples"]:
        original = prediction_dir(f["source"], sample)
        folder = prediction_dir(f["output"], sample)
        canonical = np.load(original / "canonical_depth.npy")
        metric = np.load(folder / "metric_depth.npy")
        expected, conversion = calibrated_prediction(canonical, (4, 6), calibration)
        np.testing.assert_array_equal(metric, expected)
        assert conversion["focal_native_px"] == 150 and conversion["multiplier"] == .5
        assert conversion["gt_used"] is False and conversion["applied_exactly_once"]
        pred_meta = read_json(folder / "prediction.json")
        assert pred_meta["metric_units"] == "metres" and pred_meta["gt_alignment"] is False
        assert pred_meta["source_prediction"]["canonical_sha256"] == digest_file(original / "canonical_depth.npy")
        assert (folder / "absolute_error.png").exists()
        # Last GT pixel is invalid, despite the prediction being positive.
        assert np.array_equal(np.asarray(Image.open(folder / "absolute_error.png"))[-1, -1], [150, 150, 150])
        assert digest_file(original / "canonical_depth.npy") == digest_file(folder / "canonical_depth.npy")
    result = read_json(f["output"] / "metrics_summary.json")
    assert not result["gt_alignment"] and result["regions"]["full_image"]["RMSE"] > 0
    assert result["regions"]["full_image"]["valid_pixels"] == 46
    assert result["regions"]["transparent_objects"]["valid_pixels"] == 24
    assert result["regions"]["non_transparent_regions"]["valid_pixels"] == 22
    assert (f["output"] / "contact_sheet.png").exists()
    assert len(list((f["output"] / "figures").glob("*_transparent_zoom.png"))) == 2
    assert "non_transparent_regions" in (f["output"] / "metrics_summary.csv").read_text()
    with pytest.raises(ValueError, match="nonempty"):
        cpu_runner.run(arguments(cpu_runner, f))


def test_inspection_discovers_files_but_does_not_guess_resolution(source_run):
    f = source_run
    report = inspect_calibration(f["camera_root"], input_dir=f["source"])
    assert not report["ready_for_metric"]
    assert report["candidates"][0]["valid_K"]
    assert report["candidates"][0]["dtype"] == "float32"
    assert report["actual_rgb_shapes"] == [(4, 6)]
    report = inspect_calibration(f["camera_root"], intrinsics_shape=[4, 6], input_dir=f["source"])
    assert report["ready_for_metric"]
    report = inspect_calibration(f["camera_root"], intrinsics_shape=[8, 12], input_dir=f["source"])
    assert not report["ready_for_metric"]
    # Real files are read, mask evidence is reported separately from GT validity.
    annotations = report["rgb_samples"][0]["annotations"]
    assert annotations["mask_values"] == [0, 1] and annotations["transparent_pixels"] == 12
    assert annotations["valid_gt_pixels"] == 23


@pytest.mark.parametrize("array", [np.eye(4), np.zeros((3, 4)), np.eye(3),
    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), np.full((3, 3), np.nan),
    np.diag([-300, 300, 1]), np.full((3, 3), "not intrinsics")])
def test_invalid_intrinsics_not_poses(tmp_path, array):
    path = tmp_path / "1-camIntrinsics-D435.npy"
    np.save(path, array)
    with pytest.raises(ValueError):
        validate_intrinsics(path, 1, [4, 6])


def test_camera_mismatch_pose_filename_and_discovery_ambiguity(source_run):
    f = source_run
    pose = f["camera_root"] / "T_camera2_camera1.npy"
    np.save(pose, np.diag([300, 300, 1]))
    with pytest.raises(ValueError, match="Extrinsics"):
        validate_intrinsics(pose, 1)
    l515 = f["camera_root"] / "2-camIntrinsics-L515.npy"
    np.save(l515, np.diag([300, 300, 1]))
    with pytest.raises(ValueError, match="other camera"):
        validate_intrinsics(l515, 1)
    duplicate = f["camera_root"] / "1-camIntrinsics-D435.npy"
    duplicate.write_bytes(f["K_path"].read_bytes())
    with pytest.raises(ValueError, match="Ambiguous"):
        load_calibration(f["root"], 1, intrinsics_shape=[4, 6], calibration_root=f["camera_root"])
    assert load_calibration(f["root"], 1, f["K_path"], [4, 6], f["camera_root"])


def test_actual_anisotropic_native_grid_scaling():
    K = np.array([[900., 0, 640], [0, 870., 360], [0, 0, 1]])
    canonical = np.full((322, 560), .7, np.float32)
    metric, info = metric_depth_from_canonical(canonical, K, (720, 1280))
    expected = (900 * 560 / 1280 + 870 * 322 / 720) / 2 / 300
    np.testing.assert_allclose(metric, .7 * expected, rtol=1e-6)
    assert info["multiplier"] == pytest.approx(expected)


def test_missing_grid_or_intrinsics_cannot_emit_metric(source_run, cpu_runner):
    args = arguments(cpu_runner, source_run)
    args.intrinsics_shape = None
    with pytest.raises(ValueError, match="confirmed original RGB grid"):
        cpu_runner.run(args)
    assert not source_run["output"].exists()


def test_gt_changes_metrics_not_metric_prediction(source_run, cpu_runner):
    f = source_run
    assert cpu_runner.run(arguments(cpu_runner, f)) == 0
    first = {s["sample_id"]: np.load(prediction_dir(f["output"], s) / "metric_depth.npy") for s in f["samples"]}
    before = read_json(f["output"] / "metrics_summary.json")["regions"]["full_image"]["RMSE"]
    for s in f["samples"]:
        Image.fromarray(np.full((4, 6), 2000, np.uint16)).save(Path(s["path"]).parent / "depth1-gt.png")
    args = arguments(cpu_runner, f)
    args.calibrated_output_dir = f["output"].with_name("different_GT_test")
    assert cpu_runner.run(args) == 0
    for s in f["samples"]:
        np.testing.assert_array_equal(first[s["sample_id"]], np.load(prediction_dir(args.calibrated_output_dir, s) / "metric_depth.npy"))
    assert read_json(args.calibrated_output_dir / "metrics_summary.json")["regions"]["full_image"]["RMSE"] != before


@pytest.mark.parametrize("damage", ["canonical", "RGB", "metric_label", "sample_config"])
def test_damaged_sample_is_failure_without_replacement(source_run, cpu_runner, damage):
    f = source_run
    sample = f["samples"][0]
    folder = prediction_dir(f["source"], sample)
    if damage == "canonical":
        (folder / "canonical_depth.npy").write_bytes(b"bad array")
    elif damage == "RGB":
        Path(sample["path"]).write_bytes(b"bad RGB")
    else:
        meta = read_json(folder / "prediction.json")
        if damage == "metric_label":
            meta["metric_units"] = "metres"
        else:
            meta["run_config"]["process_res_method"] = "upper_bound_crop"
        write_json(folder / "prediction.json", meta)
    before = snapshot(f["source"])
    assert cpu_runner.run(arguments(cpu_runner, f)) == 1
    assert snapshot(f["source"]) == before
    assert len(read_json(f["output"] / "inference_failures.json")) == 1
    assert len(read_json(f["output"] / "sample_manifest.json")["samples"]) == 2
    assert not (prediction_dir(f["output"], sample) / "metric_depth.npy").exists()
    assert (f["output"] / "contact_sheet.png").exists()


@pytest.mark.parametrize("field,value", [("model_id", "wrong-model"), ("da3_code_commit", "wrong-commit"),
    ("gt_alignment", True), ("checkpoint_hashes", {})])
def test_source_model_provenance_required(source_run, field, value):
    path = source_run["source"] / "run_metadata.json"
    meta = read_json(path)
    meta["model"][field] = value
    write_json(path, meta)
    with pytest.raises(ValueError):
        validate_source_run(source_run["source"])


def test_no_repeated_conversion_and_no_output_inside_source(source_run, cpu_runner):
    f = source_run
    args = arguments(cpu_runner, f)
    args.calibrated_output_dir = f["source"] / "derived"
    with pytest.raises(ValueError, match="disjoint"):
        cpu_runner.run(args)
    assert cpu_runner.run(arguments(cpu_runner, f)) == 0
    with pytest.raises(ValueError, match="already calibrated"):
        validate_source_run(f["output"])


@pytest.mark.parametrize("missing", ["mask", "GT"])
def test_safe_missing_annotations(source_run, cpu_runner, missing):
    f = source_run
    for sample in f["samples"]:
        name = "depth1-gt-mask.png" if missing == "mask" else "depth1-gt.png"
        (Path(sample["path"]).parent / name).unlink()
    assert cpu_runner.run(arguments(cpu_runner, f)) == 0
    if missing == "mask":
        report = read_json(f["output"] / "metrics_summary.json")
        assert report["regions"]["full_image"]["valid_pixels"] == 46
        assert report["regions"]["transparent_objects"]["AbsRel"] is None
        assert report["skipped"]
    else:
        assert not (f["output"] / "metrics_summary.json").exists()
        assert read_json(f["output"] / "evaluation_status.json")["status"] == "unavailable"


def test_checkpoint_and_calibrated_output_config(source_run, cpu_runner, tmp_path):
    f = source_run
    path = tmp_path / "cpu.yaml"
    path.write_text(f"input_dir: {f['source'].as_posix()}\noutput_dir: ignored_original_inference_path\n"
        f"calibrated_output_dir: {f['output'].as_posix()}\ncalibration_root: {f['camera_root'].as_posix()}\n"
        "intrinsics_shape: [4, 6]\nzoom_regions: false\n", encoding="utf-8")
    args = cpu_runner.parser().parse_args(["--config", str(path)])
    assert args.calibrated_output_dir == f["output"] and not args.zoom_regions
    assert cpu_runner.run(args) == 0
    (f["checkpoint"] / "model.safetensors").write_text("changed checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        validate_source_run(f["source"], f["checkpoint"])


def test_cpu_cli_never_imports_torch_or_official_model(source_run, tmp_path):
    f = source_run
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    # Run the complete synthetic conversion under an import guard, not just --help.
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guard(name, *a, **kw):
    if name.split('.')[0] in {'torch', 'depth_anything_3', 'huggingface_hub'}:
        raise AssertionError('Forbidden model/GPU import: ' + name)
    return original(name, *a, **kw)
builtins.__import__ = guard
script = sys.argv[1]
sys.path.insert(0, str(__import__('pathlib').Path(script).parent))
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, str(scripts / "calibrate_transcg_predictions.py"),
        "--input-dir", str(f["source"]), "--output-dir", str(f["output"]),
        "--calibration-root", str(f["camera_root"]), "--intrinsics-shape", "4", "6"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
