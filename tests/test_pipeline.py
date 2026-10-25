import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

from src.common import read_json, sample_output, digest_file
from src.datasets.booster import BoosterDataset
from src.inference.da3_zero_shot import resize_prediction

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_zero_shot import run, parser as inference_parser
from evaluate_zero_shot import evaluate, parser as eval_parser


class FakePredictor:
    """Test-only RGB-only stand-in; never reported as model inference."""
    calls = 0

    def __init__(self, *args):
        self.metadata = dict(model_id="test-double", depth_semantics="relative_raw_prediction")

    def predict(self, rgb, process_res, method):
        FakePredictor.calls += 1
        assert rgb.dtype == np.uint8 and rgb.shape == (2,3,3)
        return np.array([[.5,1.,1.],[1.,1.,1.]],np.float32), dict(
            seconds=0., native_shape=[2,3], peak_cuda_allocated_bytes=None, prediction_is_metric=0)


def test_pipeline_formats_alignment_and_resume(booster_root, tmp_path):
    output = tmp_path/"results"
    args = inference_parser().parse_args(["--data-root",str(booster_root),"--output-dir",str(output),"--max-samples","1"])
    assert run(args,FakePredictor) == 0
    ds = BoosterDataset(booster_root)
    folder = sample_output(output,"train",ds.sample_id(0))
    raw = np.load(folder/"raw.npy",allow_pickle=False)
    assert raw.dtype == np.float32 and raw.shape == (2,3)
    checksum = digest_file(folder/"raw.npy")
    calls = FakePredictor.calls
    args.resume = True
    assert run(args,FakePredictor) == 0
    assert FakePredictor.calls == calls
    eargs = eval_parser().parse_args(["--data-root",str(booster_root),"--output-dir",str(output),"--max-samples","1"])
    assert evaluate(eargs) == 0
    summary = read_json(output/"metrics_summary.json")
    assert summary["mode"] == "scale_aligned"
    assert summary["regions"]["full_image"]["RMSE"] == pytest.approx(0)
    assert summary["regions"]["target_classes_2_3"]["AbsRel"] is None
    assert digest_file(folder/"raw.npy") == checksum
    assert read_json(folder/"alignment.json")["scale"] == 2
    assert (output/"metrics_summary.csv").read_text().startswith("region,status,")
    assert (output/"metrics_per_image.csv").is_file()
    assert read_json(output/"failures.json") == dict(inference=[],evaluation=[])
    eargs.mode = "raw_metric"
    with pytest.raises(ValueError,match="not metric"):
        evaluate(eargs)


def test_failures_recorded_not_silently_skipped(booster_root, tmp_path):
    output=tmp_path/"results"
    ds = BoosterDataset(booster_root)
    ds.paths(0)["rgb"].write_bytes(b"broken image")
    args=inference_parser().parse_args(["--data-root",str(booster_root),"--output-dir",str(output)])
    assert run(args,FakePredictor) == 1
    failures=read_json(output/"inference_failures.json")
    assert failures[0]["sample_id"] == ds.sample_id(0)
    assert read_json(output/"inference_summary.json")["failed"] == 1


def test_resize_prediction_preserves_geometry_and_invalid_support():
    pred=np.array([[1,2],[3,4]],np.float32)
    resized=resize_prediction(pred,(4,6))
    assert resized.shape == (4,6) and resized.dtype == np.float32
    assert resized[0,0] == 1 and resized[-1,-1] == 4
    pred[0,0]=np.nan
    assert np.isnan(resize_prediction(pred,(4,6))[0,0])


def test_resume_rejects_corrupt_array(booster_root,tmp_path):
    output=tmp_path/"results"
    args=inference_parser().parse_args(["--data-root",str(booster_root),"--output-dir",str(output),"--max-samples","1"])
    assert run(args,FakePredictor) == 0
    ds=BoosterDataset(booster_root)
    folder=sample_output(output,"train",ds.sample_id(0))
    np.save(folder/"raw.npy",np.zeros((2,3),np.float32))
    args.resume=True
    assert run(args,FakePredictor) == 1
    assert "Corrupt" in read_json(output/"inference_failures.json")[0]["error"]


def test_outputs_cannot_be_written_into_dataset(booster_root):
    args=inference_parser().parse_args(["--data-root",str(booster_root),"--output-dir",str(booster_root/"results")])
    with pytest.raises(ValueError,match="read-only dataset"):
        run(args,FakePredictor)
