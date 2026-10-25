"""Compare real saved camera transforms with official processing; no weights/GT fit."""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path
import numpy as np

from src.common import write_json, protect_dataset, positive_int
from src.datasets.booster import BoosterDataset
from src.inference.da3_zero_shot import metric_depth_from_canonical, MODEL_ID, DA3_COMMIT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--process-res', type=positive_int, default=560)
    p.add_argument('--output', type=Path, default=Path('logs/metric_scale_validation.json'))
    args = p.parse_args()
    protect_dataset(args.data_root, args.output.parent)
    if not (args.data_root / args.split / 'index.json').is_file():
        write_json(args.output, dict(model_id=MODEL_ID, inspected_official_commit=DA3_COMMIT,
                   status='dataset_unavailable', selected=None, passed=0,
                   errors=[dict(error='Requested processed split/index.json is unavailable; no real samples checked')],
                   weights_loaded=False, model_inference_performed=False, gt_alignment=False))
        print('Dataset unavailable; no real samples checked. Diagnostic report saved.')
        return 1
    import torch
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from depth_anything_3.utils.alignment import apply_metric_scaling
    ds = BoosterDataset(args.data_root,args.split)
    processor = InputProcessor()
    entries, errors = [], []
    for i in range(len(ds)):
        try:
            rgb, meta, _ = ds.rgb_input(i)
            K = ds.calibrated_K(meta)
            images, _, official_K = processor([rgb],intrinsics=K[None],process_res=args.process_res,
                                               process_res_method='upper_bound_resize',sequential=True)
            h,w = images.shape[-2:]
            # Probe is deliberately synthetic. No checkpoint/prediction/GT is used.
            probe = np.linspace(.1,2.,h*w,dtype=np.float32).reshape(h,w)
            ours, conversion = metric_depth_from_canonical(probe,K,rgb.shape[:2])
            official = apply_metric_scaling(torch.from_numpy(probe)[None,None],official_K[None])[0,0].numpy()
            np.testing.assert_allclose(conversion['native_K'],official_K[0].numpy(),rtol=1e-6,atol=1e-5)
            np.testing.assert_allclose(ours,official,rtol=1e-6,atol=1e-6)
            entries.append(dict(sample_id=ds.sample_id(i), **conversion,
                                max_synthetic_probe_difference_m=float(np.max(np.abs(ours-official)))))
        except Exception as exc:
            errors.append(dict(sample_id=ds.sample_id(i),error=f'{type(exc).__name__}: {exc}'))
    report = dict(model_id=MODEL_ID, inspected_official_commit=DA3_COMMIT, process_res=args.process_res,
                  selected=len(ds), passed=len(entries), errors=errors, samples=entries,
                  test_kind='source-only calibration/units validation with synthetic network output probes',
                  weights_loaded=False, model_inference_performed=False, gt_alignment=False,
                  caveat='Not a GPU model test or an experimental accuracy result')
    write_json(args.output,report)
    print(f'Official calibration/scaling comparison: {len(entries)}/{len(ds)} passed; {len(errors)} errors')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
