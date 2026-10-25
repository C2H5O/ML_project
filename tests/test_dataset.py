import json
import numpy as np
from PIL import Image
import pytest
from src.datasets.booster import BoosterDataset


def test_single_sample_pairing_and_invalid_depth(booster_root):
    ds = BoosterDataset(booster_root)
    s = ds[0]
    assert s["rgb"].dtype == np.uint8 and s["rgb"].shape == (2,3,3)
    assert s["rgb"][0,0,0] == 120
    assert ds[1]["rgb"][0,0,0] == 200
    assert s["sample_id"] == "balanced/Room/camera_00"
    assert s["scene_id"] == "Room"
    assert s["depth_gt"][0,1] == 2
    assert np.array_equal(s["material_mask"], [[0,1,2],[3,3,0]])
    assert np.array_equal(s["valid_mask"], [[True,True,False],[False,False,False]])
    assert s["material_mask"][0,0] == 0 and s["valid_mask"][0,0]
    assert s["updated_K"].shape == (3,3)


def test_selection_and_no_silent_split_fallback(booster_root):
    assert len(BoosterDataset(booster_root, scenes=["Room"], subset="balanced")) == 1
    with pytest.raises(FileNotFoundError):
        BoosterDataset(booster_root, "test")
    with pytest.raises(ValueError):
        BoosterDataset(booster_root, scenes=["missing"])


def test_mismatch_rejected(booster_root):
    path = booster_root / "train/balanced/Room/camera_00/material_mask.png"
    Image.fromarray(np.zeros((3,3),np.uint8)).save(path)
    with pytest.raises(ValueError, match="size mismatch"):
        BoosterDataset(booster_root)[0]


def test_coloured_mask_rejected(booster_root):
    path = booster_root / "train/balanced/Room/camera_00/material_mask.png"
    rgb = np.zeros((2,3,3),np.uint8)
    rgb[...,1] = 2
    Image.fromarray(rgb).save(path)
    with pytest.raises(ValueError, match="colour mask"):
        BoosterDataset(booster_root)[0]


def test_missing_material_is_unavailable(booster_root):
    folder = booster_root / "train/balanced/Room/camera_00"
    meta = json.loads((folder/"meta.json").read_text())
    meta["has_material_mask"] = False
    (folder/"meta.json").write_text(json.dumps(meta))
    assert BoosterDataset(booster_root)[0]["material_mask"] is None


def test_units_and_identity_must_be_confirmed(booster_root):
    path = booster_root / "train/balanced/Room/camera_00/meta.json"
    meta = json.loads(path.read_text())
    meta["depth_unit"] = "mm"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="depth unit"):
        BoosterDataset(booster_root)[0]


def test_inference_rgb_reader_does_not_read_gt(booster_root):
    ds = BoosterDataset(booster_root)
    ds.paths(0)["depth_gt"].unlink()
    rgb, _, _ = ds.rgb_input(0)
    assert rgb.shape == (2,3,3)
    with pytest.raises(FileNotFoundError):
        ds[0]
