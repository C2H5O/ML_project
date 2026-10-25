"""Single-image official DA3 zero-shot inference; no GT/model input coupling."""
import _bootstrap  # noqa: F401
import argparse
import random
from pathlib import Path
import numpy as np

from src.common import (data_arguments, selected_indices, sample_output, write_json, read_json,
                        write_npy, digest_file, positive_int, protect_dataset)
from src.datasets.booster import BoosterDataset
from src.inference.da3_zero_shot import DA3Predictor, MODELS, resize_prediction


def run(args, predictor_factory=DA3Predictor):
    protect_dataset(args.data_root, args.output_dir)
    ds = BoosterDataset(args.data_root, args.split, args.scene, args.subset)
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    failures, records = [], []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        predictor = predictor_factory(args.model_id, args.checkpoint, args.device, args.precision, args.revision)
    except Exception as exc:
        write_json(args.output_dir / "inference_failures.json", [dict(stage="model_load", error=f"{type(exc).__name__}: {exc}")])
        raise
    config = dict(**predictor.metadata, process_res=args.process_res,
                  process_res_method=args.process_res_method, seed=args.seed, split=args.split,
                  single_image=True, gt_input=False, infer_gs=False)
    manifest = args.output_dir / "run_metadata.json"
    if manifest.exists() and read_json(manifest) != config:
        raise ValueError("Output directory belongs to a different run config; choose another --output-dir")
    write_json(manifest, config)
    for i in selected_indices(ds, args.max_samples):
        sid = ds.sample_id(i)
        out = sample_output(args.output_dir, args.split, sid)
        try:
            rgb, meta, paths = ds.rgb_input(i)
            input_hash = digest_file(paths["rgb"])
            success = out / "prediction.json"
            if args.resume and success.exists():
                previous = read_json(success)
                if previous["sample_id"] == sid and previous["input_sha256"] == input_hash and previous["run_config"] == config:
                    # Validate format and hashes, not merely existence.
                    for name, shape in [("raw_native.npy", previous["native_shape"]), ("raw.npy", list(rgb.shape[:2]))]:
                        array = np.load(out / name, allow_pickle=False)
                        if array.dtype != np.float32 or list(array.shape) != shape or digest_file(out / name) != previous["output_sha256"][name]:
                            raise ValueError("Corrupt resumed output; rerun without --resume")
                    records.append(dict(sample_id=sid, status="resumed"))
                    print(f"RESUMED {sid}", flush=True)
                    continue
                raise ValueError("Resume input or config mismatch; use a new output directory")
            native, timing = predictor.predict(rgb, args.process_res, args.process_res_method)
            raw = resize_prediction(native, rgb.shape[:2])
            # Remove stale success marker before replacing either array.
            success.unlink(missing_ok=True)
            write_npy(out / "raw_native.npy", native)
            write_npy(out / "raw.npy", raw)
            write_json(success, dict(sample_id=sid, split=args.split, run_config=config,
                                     input_sha256=input_hash, input_shape=list(rgb.shape[:2]),
                                     original_size=meta["original_size"], **timing,
                                     output_sha256={name: digest_file(out / name) for name in ["raw_native.npy", "raw.npy"]},
                                     raw_units="metres" if "METRIC" in args.model_id else "model-relative units",
                                     resize="bilinear prediction to saved RGB/GT size; GT and masks unchanged"))
            records.append(dict(sample_id=sid, status="success", **timing))
            print(f"OK {sid} {timing['seconds']:.3f}s", flush=True)
        except Exception as exc:
            failures.append(dict(sample_id=sid, stage="inference", error=f"{type(exc).__name__}: {exc}"))
            print(f"FAILED {sid}: {exc}", flush=True)
        finally:
            write_json(args.output_dir / "inference_failures.json", failures)
            write_json(args.output_dir / "inference_summary.json", dict(selected=len(selected_indices(ds, args.max_samples)), records=records, failed=len(failures)))
    return 1 if failures else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    data_arguments(p)
    p.add_argument("--model-id", choices=MODELS, default=MODELS[0])
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--revision", help="HF model revision; resolved to immutable SHA")
    p.add_argument("--output-dir", type=Path, default=Path("results/zero_shot"))
    p.add_argument("--process-res", type=positive_int, default=560)
    p.add_argument("--process-res-method", choices=["upper_bound_resize", "lower_bound_resize"], default="upper_bound_resize")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
