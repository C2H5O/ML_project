"""Synthetic fixtures only. No real DA3 model, downloads, GPU or dataset writes."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.common import read_json
from src.datasets.transcg import (discover_root, scan_rgb, sample_rgb, read_annotations,
    load_calibration, calibrated_prediction, prediction_dir)
from src.visualization.transcg import colour_depth


def make_rgb(root, scene=1, frame="0", camera=1):
    folder = root / f"scene{scene}" / str(frame)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 6, 3), 80, np.uint8)).save(folder / f"rgb{camera}.png")
    return folder


@pytest.mark.parametrize("layout", [".", "data", "transcg/data", "transcg"])
def test_layouts(tmp_path, layout):
    actual = tmp_path / layout
    make_rgb(actual, frame="nested/17")
    root, samples, report = scan_rgb(tmp_path, 1, 1)
    assert root == actual.resolve()
    assert report["total_rgb"] == 1
    assert samples[0]["frame_id"] == "nested/17"
    assert samples[0]["sample_id"] == "scene1/nested/17/camera1"
    assert samples[0]["camera_name"] == "D435"


def test_rgb_filter_and_bad_files(tmp_path):
    folder = make_rgb(tmp_path)
    make_rgb(tmp_path, camera=2)
    for name in ["depth1.png", "depth1-gt.png", "depth1-gt-mask.png", "rgb1-mask.png"]:
        Image.fromarray(np.ones((4, 6), np.uint8)).save(folder / name)
    bad = tmp_path / "scene1" / "bad"
    bad.mkdir()
    (bad / "rgb1.png").write_text("not PNG")
    make_rgb(tmp_path, scene=11)
    _, candidates, report = scan_rgb(tmp_path)
    assert len(candidates) == 1 and candidates[0]["camera_id"] == 1
    assert report["per_scene"]["scene2"] == 0
    assert len(report["rejected_rgb"]) == 1
    assert len(scan_rgb(tmp_path, camera=2)[1]) == 1


def test_sampling(tmp_path):
    for scene in range(1, 11):
        for frame in range(3):
            make_rgb(tmp_path, scene, frame)
    _, candidates, _ = scan_rgb(tmp_path)
    a = sample_rgb(candidates)
    assert a == sample_rgb(candidates, seed=42)
    assert a != sample_rgb(candidates, seed=43)
    assert len({s["path"] for s in a}) == 20
    per_scene = sample_rgb(candidates, sampling="per_scene")
    assert all(sum(s["scene_id"] == f"scene{i}" for s in per_scene) == 2 for i in range(1, 11))
    with pytest.raises(ValueError):
        sample_rgb(candidates, 31)
    with pytest.raises(ValueError):
        sample_rgb(candidates, 21, sampling="per_scene")
    with pytest.raises(ValueError):
        sample_rgb(candidates, 20, sampling="per_scene", scenes=["scene1", "scene2", "scene3", "scene4", "scene9", "scene10", "scene7", "scene8", "scene5", "scene_missing"])


def test_ambiguous_and_invalid_scenes(tmp_path):
    make_rgb(tmp_path)
    make_rgb(tmp_path / "data")
    with pytest.raises(ValueError, match="unambiguous"):
        discover_root(tmp_path)
    with pytest.raises(ValueError, match="scene1"):
        discover_root(tmp_path, 0, 11)


@pytest.mark.parametrize("camera,stored", [(1, 1000), (2, 4000)])
def test_annotations_safe_and_units(tmp_path, camera, stored):
    folder = make_rgb(tmp_path, camera=camera)
    sample = scan_rgb(tmp_path, camera=camera)[1][0]
    # A valid sensor depth file MUST NOT act as missing GT.
    Image.fromarray(np.full((4, 6), stored, np.uint16)).save(folder / f"depth{camera}.png")
    gt, mask, info = read_annotations(sample, (4, 6))
    assert gt is None and mask is None and not info["gt_available"]
    Image.fromarray(np.full((4, 6), stored, np.uint16)).save(folder / f"depth{camera}-gt.png")
    Image.fromarray(np.ones((4, 6), np.uint8)).save(folder / f"depth{camera}-gt-mask.png")
    gt, mask, info = read_annotations(sample, (4, 6))
    assert np.all(gt == 1) and mask.all() and info["gt_units"] == "metres"
    Image.fromarray(np.ones((4, 6), np.uint8) * 3).save(folder / f"depth{camera}-gt-mask.png")
    assert read_annotations(sample, (4, 6))[1] is None
    assert read_annotations(sample, (2, 3))[0] is None


def test_calibration_required_and_grid_scaled(tmp_path):
    canonical = np.full((2, 3), 2, np.float32)
    assert calibrated_prediction(canonical, (4, 6), None) == (None, None)
    assert load_calibration(tmp_path, 1) is None
    with pytest.raises(ValueError, match="other camera"):
        load_calibration(tmp_path, 1, tmp_path / "2-camIntrinsics-L515.npy", [4, 6])
    path = tmp_path / "K.npy"
    np.save(path, np.diag([300., 300., 1.]))
    with pytest.raises(ValueError, match="intrinsics-shape"):
        load_calibration(tmp_path, 1, path)
    calibration = load_calibration(tmp_path, 1, path, [4, 6])
    metric, conversion = calibrated_prediction(canonical, (4, 6), calibration)
    assert np.all(metric == 1)  # actual output grid halves focal
    assert conversion["focal_native_px"] == 150
    assert not conversion["gt_used"]
    with pytest.raises(ValueError, match="dimensions"):
        calibrated_prediction(canonical, (8, 12), calibration)
    np.save(path, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="intrinsics"):
        load_calibration(tmp_path, 1, path, [4, 6])


def test_prediction_path(tmp_path):
    make_rgb(tmp_path)
    sample = scan_rgb(tmp_path)[1][0]
    out = prediction_dir(tmp_path / "output", sample)
    assert out.parent == tmp_path / "output" / "predictions" / "scene1"
    assert out == prediction_dir(tmp_path / "output", sample)


@pytest.fixture
def runner():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("transcg_runner_test", scripts / "run_transcg_zero_shot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(scripts))


class SyntheticPredictor:
    calls = 0
    loads = 0

    def __init__(self, **kwargs):
        import os
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert kwargs["checkpoint"].is_dir()
        type(self).loads += 1
        self.metadata = {"model_id": "synthetic-test-only", "depth_semantics": "raw_metric_metres"}

    def predict_canonical(self, rgb, process_res, method):
        type(self).calls += 1
        assert rgb.dtype == np.uint8 and rgb.shape == (4, 6, 3)
        assert process_res == 560 and method == "upper_bound_resize"
        return np.full((2, 3), 2, np.float32), dict(seconds=0.01, native_shape=[2, 3],
            peak_cuda_allocated_bytes=None, prediction_is_metric=None)


def make_args(runner, tmp_path, extra=()):
    root = tmp_path / "dataset"
    for i in range(3):
        make_rgb(root, frame=i)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_text("SYNTHETIC TEST ONLY")
    return runner.parser().parse_args(["--data-root", str(root), "--checkpoint", str(checkpoint),
        "--output-dir", str(tmp_path / "outputs"), "--num-samples", "2", "--scene-end", "1", *extra])


def test_pipeline_canonical_resume_and_protection(tmp_path, runner):
    args = make_args(runner, tmp_path)
    SyntheticPredictor.calls = SyntheticPredictor.loads = 0
    assert runner.run(args, SyntheticPredictor) == 0
    assert SyntheticPredictor.calls == 2 and SyntheticPredictor.loads == 1
    output = args.output_dir
    manifest = read_json(output / "sample_manifest.json")
    original = (output / "sample_manifest.json").read_bytes()
    assert len(manifest["samples"]) == 2
    assert not (output / "metrics_summary.json").exists()
    assert (output / "contact_sheet.png").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 2
    for sample in manifest["samples"]:
        folder = prediction_dir(output, sample)
        assert (folder / "canonical_depth.npy").is_file()
        assert not (folder / "metric_depth.npy").exists()
        assert (folder / "predicted_depth.png").is_file()
        assert read_json(folder / "prediction.json")["metric_units"] is None
    assert read_json(output / "run_metadata.json")["model"]["depth_semantics"] == "canonical network output"
    with pytest.raises(ValueError, match="Existing"):
        runner.run(args, SyntheticPredictor)
    args.resume = True
    assert runner.run(args, SyntheticPredictor) == 0
    assert SyntheticPredictor.calls == 2
    assert (output / "sample_manifest.json").read_bytes() == original
    # Destroy a selected input AFTER selection: no replacement, explicit failure.
    Path(manifest["samples"][0]["path"]).write_bytes(b"broken")
    assert runner.run(args, SyntheticPredictor) == 1
    assert SyntheticPredictor.calls == 2
    assert len(read_json(output / "inference_failures.json")) == 1
    assert (output / "sample_manifest.json").read_bytes() == original
    args.process_res = 448
    with pytest.raises(ValueError, match="configuration"):
        runner.run(args, SyntheticPredictor)


def test_pipeline_metric_only_with_calibration_and_gt(tmp_path, runner):
    args = make_args(runner, tmp_path)
    path = tmp_path / "K.npy"
    np.save(path, np.diag([300., 300., 1.]))
    args.intrinsics, args.intrinsics_shape = path, [4, 6]
    for folder in args.data_root.glob("scene1/*"):
        Image.fromarray(np.full((4, 6), 1000, np.uint16)).save(folder / "depth1-gt.png")
        Image.fromarray(np.ones((4, 6), np.uint8)).save(folder / "depth1-gt-mask.png")
    assert runner.run(args, SyntheticPredictor) == 0
    metrics = read_json(args.output_dir / "metrics_summary.json")
    assert metrics["full_image"]["RMSE"] == 0
    assert metrics["transparent_objects"]["delta1"] == 1
    args.resume = True
    sample = read_json(args.output_dir / "sample_manifest.json")["samples"][0]
    (prediction_dir(args.output_dir, sample) / "canonical_depth.npy").write_bytes(b"corrupt")
    assert runner.run(args, SyntheticPredictor) == 1
    assert read_json(args.output_dir / "inference_failures.json")


def test_nonempty_output_and_no_download_fallback(tmp_path, runner):
    args = make_args(runner, tmp_path)
    args.output_dir.mkdir()
    sentinel = args.output_dir / "metrics_summary.json"
    sentinel.write_text("existing Booster metrics")
    with pytest.raises(ValueError, match="nonempty"):
        runner.run(args, SyntheticPredictor)
    assert sentinel.read_text() == "existing Booster metrics"
    args.checkpoint = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match="downloads are disabled"):
        runner.run(args, SyntheticPredictor)


def test_model_failure_and_incomplete_contact_sheet(tmp_path, runner):
    args = make_args(runner, tmp_path)
    def fail(**kwargs):
        raise RuntimeError("synthetic model load failure")
    assert runner.run(args, fail) == 1
    assert read_json(args.output_dir / "inference_failures.json")[0]["stage"] == "model_load"
    assert len(read_json(args.output_dir / "visualization_summary.json")["failures"]) == 2
    assert (args.output_dir / "contact_sheet.png").exists()


def test_zero_error_colour_is_valid():
    assert np.array_equal(np.asarray(colour_depth(np.zeros((1, 1)), 0, 1, "magma"))[0, 0], [0, 0, 3])


def test_canonical_wrapper_does_not_require_or_convert_intrinsics(monkeypatch):
    import types
    import torch
    import src.inference.da3_zero_shot as adapter
    def forbidden_conversion(*args):
        raise AssertionError("Canonical inference must not convert units")
    monkeypatch.setattr(adapter, "metric_depth_from_canonical", forbidden_conversion)
    class API:
        def inference(self, **kwargs):
            assert set(kwargs) == {"image", "process_res", "process_res_method", "infer_gs", "export_dir"}
            assert len(kwargs["image"]) == 1
            return types.SimpleNamespace(depth=np.ones((1, 2, 3), np.float32), is_metric=0,
                processed_images=np.zeros((1, 2, 3, 3), np.uint8))
    predictor = adapter.DA3Predictor.__new__(adapter.DA3Predictor)
    predictor.device, predictor.model = torch.device("cpu"), API()
    depth, info = predictor.predict_canonical(np.zeros((4, 6, 3), np.uint8))
    assert depth.shape == (2, 3) and "metric_conversion" not in info


def test_yaml_config_pipeline_and_cli_overrides(tmp_path, runner):
    import yaml
    from src.transcg_config import PROJECT_ROOT
    original = make_args(runner, tmp_path)
    config_path = tmp_path / "run.yaml"
    config = dict(data_root=str(original.data_root), checkpoint=str(original.checkpoint),
        output_dir=str(original.output_dir), scene_end=1, num_samples=2, process_res=560,
        precision="bf16", resume=False, intrinsics=None, intrinsics_shape=None)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args = runner.parser().parse_args(["--config", str(config_path)])
    assert args.data_root == original.data_root and args.checkpoint == original.checkpoint
    assert runner.run(args, SyntheticPredictor) == 0
    assert runner.run(runner.parser().parse_args(["--config", str(config_path), "--resume"]), SyntheticPredictor) == 0
    config.update(resume=True, num_samples=8, output_dir="results/config-test-only")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    # Relative config paths are stable even outside the repository.
    args = runner.parser().parse_args(["--config", str(config_path), "--num-samples", "3", "--no-resume"])
    assert args.num_samples == 3 and not args.resume
    assert args.output_dir == PROJECT_ROOT / "results/config-test-only"


@pytest.mark.parametrize("text", [
    "num_sampls: 20", "num_samples: 0", "num_samples: true", "resume: 'false'",
    "camera: 3", "precision: bfloat16", "intrinsics_shape: [720]",
    "intrinsics_shape: [720, -1]", "depth_max: .nan", "data_root: null",
    "scene_start: 10\nscene_end: 1", "[not, a, mapping]", "data_root: [broken",
])
def test_invalid_yaml_fails_before_inference(tmp_path, runner, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        runner.parser().parse_args(["--config", str(path), "--data-root", str(tmp_path)])
    assert exc.value.code == 2


def test_shared_yaml_for_inspection_and_visualization(tmp_path):
    from src.transcg_config import TransCGArgumentParser
    path = tmp_path / "common.yaml"
    path.write_text("data_root: ./data\ncheckpoint: ./checkpoints/DA3METRIC-LARGE\n"
                    "num_samples: 20\noutput_dir: ./results/transcg_zero_shot\ndepth_max: 3.0\nresume: true\n")
    p = TransCGArgumentParser()
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--depth-max", type=float)
    args = p.parse_args(["--config", str(path), "--depth-max", "4"])
    assert args.output_dir.name == "transcg_zero_shot" and args.depth_max == 4


def test_delivered_server_config(runner):
    from src.transcg_config import PROJECT_ROOT
    args = runner.parser().parse_args(["--config", str(PROJECT_ROOT / "configs/transcg_zero_shot.yaml")])
    assert args.checkpoint.as_posix() == "/public/home/2024141520249/Documents/Projects/ML_pro/ML_project/checkpoints/DA3METRIC-LARGE"
    assert args.data_root.as_posix() == "/public/home/2024141520249/Documents/Projects/ML_pro/data"
    assert args.intrinsics is None and args.intrinsics_shape is None
    assert args.num_samples == 20 and args.process_res == 560 and args.precision == "bf16"
