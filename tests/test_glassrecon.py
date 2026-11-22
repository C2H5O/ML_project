import numpy as np
import pytest
from PIL import Image

from scripts.run_glassrecon_zero_shot import score_regions, summarize
from src.common import read_json, write_json
from src.datasets.glassrecon import SPLIT_MANIFEST, Sample, read_frame


def test_fixed_split_has_disjoint_buildings_and_difficulty_roster():
    manifest = read_json(SPLIT_MANIFEST)
    buildings = [building for split in manifest["splits"].values() for building in split]
    ids = {
        split: [sample_id for samples in by_building.values() for sample_id in samples]
        for split, by_building in manifest["splits"].items()
    }
    all_ids = [sample_id for samples in ids.values() for sample_id in samples]
    hard = set(manifest["difficulty"]["hard_ids"])
    assert len(buildings) == len(set(buildings)) == 26
    assert len(all_ids) == len(set(all_ids)) == 917
    assert {split: len(samples) for split, samples in ids.items()} == {
        "train": 642, "validation": 137, "test": 138,
    }
    assert hard <= set(all_ids)
    assert len(hard) == 316


def test_gt_units_camera_and_regional_metrics(tmp_path):
    paths = {key: tmp_path / filename for key, filename in {
        "rgb": "sample.jpg", "mask": "sample.png", "gt": "sample.npy", "intrinsics": "sample.json",
    }.items()}
    Image.new("RGB", (3, 2)).save(paths["rgb"])
    mask = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8)
    Image.fromarray(mask).save(paths["mask"])
    np.save(paths["gt"], np.array([[0, 4000, 0], [0, 8000, 0]], dtype=np.uint16))
    write_json(paths["intrinsics"], {
        "height": 2, "width": 3, "depth_scale": 0.00025, "fx": 3, "fy": 4, "ppx": 1.5, "ppy": 1,
    })
    frame = read_frame(Sample("sample", "building", "easy", paths))
    np.testing.assert_array_equal(frame.gt_m, [[0, 1, 0], [0, 2, 0]])
    np.testing.assert_array_equal(frame.camera, [[3, 0, 1.5], [0, 4, 1], [0, 0, 1]])

    prediction = np.array([[1, 1.25, 1], [1, np.nan, 1]])
    full, glass, non_glass = score_regions(prediction, frame.gt_m, frame.glass)
    assert full["valid_pixels"] == 1
    assert glass["gt_coverage"] == glass["valid_coverage"] == pytest.approx(1 / 3)
    assert (glass["AbsRel"], glass["RMSE"], glass["delta1"]) == (0.25, 0.25, 0)
    assert non_glass["status"] == "no_valid_prediction"
    assert non_glass["AbsRel"] is None

    missing_gt = np.zeros((2, 3), dtype=np.float32)
    assert list(score_regions(prediction, missing_gt, frame.glass))[1]["status"] == "no_valid_gt"


def test_summary_weights_each_available_image_equally():
    glass = np.ones((1, 4), dtype=bool)
    sparse = np.array([[1, 0, 0, 0]])
    dense = np.ones((1, 4))
    rows = [
        dict(difficulty=group, **region)
        for group, pred, gt in [("hard", sparse * 2, sparse), ("easy", dense, dense)]
        for region in score_regions(pred, gt, glass)
    ]
    all_glass = next(r for r in summarize(rows) if (r["difficulty"], r["region"]) == ("all", "glass"))
    assert all_glass["images_evaluated"] == 2
    assert all_glass["valid_pixels"] == 5
    assert all_glass["valid_coverage"] == 5 / 8
    assert all_glass["AbsRel"] == all_glass["RMSE"] == all_glass["delta1"] == 0.5
