"""Read-only TransCG inspection. No torch, checkpoint loading or inference."""
import _bootstrap  # noqa: F401
import json
from pathlib import Path

from PIL import Image
import numpy as np
from src.datasets.transcg import scan_rgb, read_annotations
from src.transcg_config import TransCGArgumentParser


def main():
    p = TransCGArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--camera", type=int, choices=[1, 2], default=1)
    p.add_argument("--scene-start", type=int, default=1)
    p.add_argument("--scene-end", type=int, default=10)
    args = p.parse_args()
    root, candidates, report = scan_rgb(args.data_root, args.scene_start, args.scene_end, args.camera)
    report["resolved_root"] = str(root)
    report["calibration_available_without_grid_confirmation"] = False
    report["intrinsics_files"] = [str(p) for p in (root / "camera_intrinsics").glob("*.npy")]
    if candidates:
        sample = candidates[0]
        _, _, report["annotation_example"] = read_annotations(sample, sample["input_shape"])
        report["file_examples"] = []
        for path in sorted(Path(sample["path"]).parent.glob("*.png")):
            with Image.open(path) as image:
                array = np.asarray(image)
                report["file_examples"].append(dict(name=path.name, mode=image.mode, shape=list(array.shape),
                    dtype=str(array.dtype), minimum=int(array.min()), maximum=int(array.max())))
    print(json.dumps(report, indent=2))
    print("Only RGB is model input. GT is *-gt.png; depthN.png is raw sensor depth.")
    print("D435 stored depth / 1000 = m; L515 / 4000 = m (official TransCG convention).")
    print("Without confirmed camera intrinsics AND their pixel grid: canonical-depth inference only; no trustworthy Raw Metric Evaluation.")


if __name__ == "__main__":
    main()
