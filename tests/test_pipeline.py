import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

from src.common import read_json, sample_output, digest_file
from src.datasets.booster import BoosterDataset
from src.inference.da3_zero_shot import resize_prediction, metric_depth_from_canonical, MODEL_ID, METRIC_PROTOCOL

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_zero_shot import run, parser as inference_parser
from evaluate_zero_shot import evaluate, parser as eval_parser


class FakePredictor:
    """Test-only RGB-only stand-in; never reported as model inference."""
    calls = 0

    def __init__(self, *args):
        self.metadata = dict(model_id=MODEL_ID, test_double=True, depth_semantics="raw_metric_metres",
                             metric_protocol=METRIC_PROTOCOL, gt_alignment=False)

    def predict(self, rgb, K, process_res, method):
        FakePredictor.calls += 1
        assert rgb.dtype == np.uint8 and rgb.shape == (2,3,3)
        canonical = np.array([[.5,1.,1.],[1.,1.,1.]],np.float32)
        metric, conversion = metric_depth_from_canonical(canonical, K, rgb.shape[:2])
        return canonical, metric, dict(seconds=0., native_shape=[2,3], metric_conversion=conversion,
            peak_cuda_allocated_bytes=None, prediction_is_metric=0)


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
    assert summary["mode"] == "raw_metric" and summary["gt_alignment"] is False
    assert summary["regions"]["full_image"]["RMSE"] == pytest.approx(0)
    assert summary["regions"]["target_classes_2_3"]["AbsRel"] is None
    assert digest_file(folder/"raw.npy") == checksum
    assert not (folder/"alignment.json").exists()
    assert not (folder/"scale_aligned.npy").exists()
    assert read_json(folder/"prediction.json")["metric_conversion"]["multiplier"] == 2
    assert (output/"metrics_summary.csv").read_text().startswith("region,status,")
    assert (output/"metrics_per_image.csv").is_file()
    assert read_json(output/"failures.json") == dict(inference=[],evaluation=[])
    from src.common import write_json
    config = read_json(output/"run_metadata.json")
    config["depth_semantics"] = "relative_raw_prediction"
    write_json(output/"run_metadata.json",config)
    with pytest.raises(ValueError,match="calibrated"):
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


def test_gt_changes_errors_without_changing_predictions_or_refitting_scale(booster_root,tmp_path):
    output=tmp_path/'metric'
    args=inference_parser().parse_args(['--data-root',str(booster_root),'--output-dir',str(output),'--max-samples','1'])
    assert run(args,FakePredictor) == 0
    ds=BoosterDataset(booster_root)
    out=sample_output(output,'train',ds.sample_id(0))
    raw_hash=digest_file(out/'raw.npy')
    path=ds.paths(0)['depth_gt']
    gt=np.load(path)
    gt[0,:2] *= 10
    np.save(path,gt)
    eargs=eval_parser().parse_args(['--data-root',str(booster_root),'--output-dir',str(output),'--max-samples','1'])
    assert evaluate(eargs) == 0
    result=read_json(output/'metrics_summary.json')['regions']['full_image']
    assert result['AbsRel'] == pytest.approx(.9)
    assert result['delta1'] == 0
    assert digest_file(out/'raw.npy') == raw_hash
    assert not (out/'scale_aligned.npy').exists()


def test_double_metric_scaling_is_rejected_even_with_updated_array_checksum(booster_root,tmp_path):
    from src.common import write_json
    output=tmp_path/'metric'
    args=inference_parser().parse_args(['--data-root',str(booster_root),'--output-dir',str(output),'--max-samples','1'])
    assert run(args,FakePredictor) == 0
    ds=BoosterDataset(booster_root)
    out=sample_output(output,'train',ds.sample_id(0))
    path=out/'raw_native.npy'
    np.save(path,np.load(path)*2)
    meta=read_json(out/'prediction.json')
    meta['output_sha256']['raw_native.npy']=digest_file(path)
    write_json(out/'prediction.json',meta)
    eargs=eval_parser().parse_args(['--data-root',str(booster_root),'--output-dir',str(output),'--max-samples','1'])
    assert evaluate(eargs) == 1
    assert 'possible missing/double scaling' in read_json(output/'failures.json')['evaluation'][0]['error']


def test_raw_metric_figure_does_not_change_predictions_or_use_alignment(booster_root,tmp_path,monkeypatch):
    import src.evaluation.metrics as metrics
    from src.visualization.depth_visualizer import save_panel
    from PIL import Image
    monkeypatch.setattr(metrics,'align_scale',lambda *args: (_ for _ in ()).throw(AssertionError('GT alignment forbidden')))
    sample=BoosterDataset(booster_root)[0]
    raw=np.ones((2,3),np.float32)
    original=raw.copy()
    output=save_panel(sample,raw,tmp_path/'raw_metric.png',dpi=100)
    with Image.open(output) as image:
        assert image.size == (1600,900)
    np.testing.assert_array_equal(raw,original)


def test_metre_label_cannot_hide_wrong_raw_grid_array(booster_root,tmp_path):
    from src.common import write_json
    output=tmp_path/'metric'
    args=inference_parser().parse_args(['--data-root',str(booster_root),'--output-dir',str(output),'--max-samples','1'])
    assert run(args,FakePredictor) == 0
    ds=BoosterDataset(booster_root)
    out=sample_output(output,'train',ds.sample_id(0))
    path=out/'raw.npy'
    np.save(path,np.load(path)*2)
    meta=read_json(out/'prediction.json')
    meta['output_sha256']['raw.npy']=digest_file(path)
    write_json(out/'prediction.json',meta)
    eargs=eval_parser().parse_args(['--data-root',str(booster_root),'--output-dir',str(output),'--max-samples','1'])
    assert evaluate(eargs) == 1
    assert 'does not match converted native depth' in read_json(output/'failures.json')['evaluation'][0]['error']
