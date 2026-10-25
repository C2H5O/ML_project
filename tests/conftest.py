import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def booster_root(tmp_path):
    root = tmp_path / "booster"
    split = root / "train"
    records = []
    for subset in ["balanced", "unbalanced"]:
        sid = f"{subset}/Room/camera_00"
        folder = split / sid
        folder.mkdir(parents=True)
        rgb = np.zeros((2, 3, 3), np.uint8)
        rgb[..., 0] = 120 if subset == "balanced" else 200
        depth = np.array([[1, 2, 0], [np.nan, np.inf, -1]], np.float32)
        Image.fromarray(rgb).save(folder / "rgb.png")
        np.save(folder / "depth_gt.npy", depth)
        for name, arr in [("valid_mask", np.array([[255,255,255],[255,255,255]],np.uint8)),
                          ("mask", np.full((2,3), 255,np.uint8)),
                          ("material_mask", np.array([[0,1,2],[3,3,0]],np.uint8))]:
            # Match actual stored RGB replicated-mask encoding.
            Image.fromarray(np.repeat(arr[...,None],3,axis=2)).save(folder / f"{name}.png")
        files = dict(rgb="rgb.png", depth_gt="depth_gt.npy", valid_mask="valid_mask.png",
                     mask="mask.png", material_mask="material_mask.png", meta="meta.json")
        K = np.diag([600.,600.,1.]).tolist()
        meta = dict(sample_id=sid, split="train", depth_unit="m", processed_size=[2,3],
                    original_size=[2,3], K=K, K_original=K,
                    transform=dict(resized_size=[2,3],crop_x=0,crop_y=0,scale_x=1.,scale_y=1.),
                    has_material_mask=True, files=files)
        (folder/"meta.json").write_text(json.dumps(meta))
        records.append(dict(sample_id=sid, dir=sid, meta=f"{sid}/meta.json"))
    (split/"index.json").write_text(json.dumps(dict(schema_version=1, dataset="booster", split="train",
                                                  num_samples=2, samples=records)))
    return root
