"""Offline RGB-only single-image inference. Run only on the intended server."""
from pathlib import Path
import _bootstrap
from src.common import (read_json, write_json, write_npy, digest_file, sample_output,
                        protect_dataset, positive_int)
from src.datasets.cleargrasp import (SPLITS, discover, select_samples, calibration,
                                     read_rgb, path_for)
from src.inference.da3_zero_shot import DA3Predictor, metric_depth_from_canonical, DA3_COMMIT
from src.evaluation.cleargrasp import saved_prediction, evaluate_saved
from src.visualization.cleargrasp import visualize_saved
from src.cleargrasp_config import ClearGraspArgumentParser, print_config
from evaluate_cleargrasp import protocol_arguments


def parser():
    p = ClearGraspArgumentParser(description=__doc__, required_paths=('data_root', 'checkpoint', 'output_dir'))
    p.add_argument('--split', choices=SPLITS, default='real-test')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--max-samples', type=positive_int, default=20)
    g.add_argument('--all', action='store_true')
    p.add_argument('--process-res', type=positive_int, default=560)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--precision', choices=('bf16', 'fp16', 'fp32'), default='bf16')
    p.add_argument('--depth-max', type=float, default=2.)
    p.add_argument('--error-max', type=float, default=0.5)
    p.add_argument('--skip-evaluation', action='store_true')
    p.add_argument('--skip-visualization', action='store_true')
    protocol_arguments(p)
    return p


def run(args):
    protect_dataset(args.data_root, args.output_dir)
    checkpoint = args.checkpoint.resolve()
    names = ('config.json', 'model.safetensors')
    if not checkpoint.is_dir() or not all((checkpoint / n).is_file() for n in names):
        raise FileNotFoundError('Local checkpoint must contain config.json and model.safetensors; downloads prohibited')
    manifest = discover(args.data_root)
    root = Path(manifest['data_root'])
    protect_dataset(root, args.output_dir)
    selected = select_samples(manifest, args.split, args.max_samples, args.seed, args.all)
    # Bind resume to RGB bytes and calibrated K; annotations never enter model inference.
    for s in selected:
        rgb = read_rgb(path_for(root, s, 'rgb'))
        s['input_sha256'] = digest_file(path_for(root, s, 'rgb'))
        s['input_shape'] = list(rgb.shape[:2])
        try:
            s['calibration'] = calibration(root, s, rgb.shape[:2])
            s['calibration_warning'] = None
        except Exception as exc:
            s['calibration'] = None
            s['calibration_warning'] = str(exc)
    config = dict(model_id='depth-anything/DA3METRIC-LARGE', da3_commit=DA3_COMMIT,
        checkpoint_hashes={n: digest_file(checkpoint / n) for n in names},
        process_res=args.process_res, process_res_method='upper_bound_resize', precision=args.precision,
        device=args.device, split=args.split, seed=args.seed, offline=True, gt_input=False,
        selection=selected, metric_protocol='mean(fx_native, fy_native)/300 exactly once')
    output = args.output_dir
    metadata_path = output / 'run_metadata.json'
    if metadata_path.exists():
        old = read_json(metadata_path)
        if old['run_config'] != config:
            raise ValueError('Existing run differs in inputs/checkpoint/config; choose a new output directory')
    else:
        # Avoid adopting unrelated prediction files without a run identity.
        if (output / 'predictions').exists():
            raise ValueError('Predictions exist without run metadata; choose a new output directory')
        write_json(metadata_path, dict(status='prepared', run_config=config))
    manifest['samples'] = selected
    write_json(output / 'sample_manifest.json', manifest)
    records, failures, predictor = [], [], None
    for s in selected:
        folder = sample_output(output, s['split'], s['sample_id'])
        try:
            if (folder / 'prediction.json').exists():
                _, _, meta = saved_prediction(output, s)
                if meta['run_config'] != config or meta['input_sha256'] != s['input_sha256']:
                    raise ValueError('Resume provenance differs')
                records.append(dict(sample_id=s['sample_id'], status='resumed', seconds=meta['seconds'],
                                    peak_cuda_allocated_bytes=meta['peak_cuda_allocated_bytes']))
                print('RESUMED', s['sample_id'], flush=True)
                continue
            if predictor is None:
                predictor = DA3Predictor(checkpoint=checkpoint, device=args.device, precision=args.precision)
                write_json(metadata_path, dict(status='running', run_config=config, predictor=predictor.metadata))
            rgb = read_rgb(path_for(root, s, 'rgb'))
            canonical, timing = predictor.predict_canonical(rgb, args.process_res, 'upper_bound_resize')
            metric, conversion = None, None
            if s['calibration'] is not None:
                metric, conversion = metric_depth_from_canonical(canonical, s['calibration']['K'], rgb.shape[:2])
            write_npy(folder / 'canonical_depth.npy', canonical)
            outputs = ['canonical_depth.npy']
            if metric is not None:
                write_npy(folder / 'metric_depth.npy', metric)
                outputs.append('metric_depth.npy')
            write_json(folder / 'prediction.json', dict(sample_id=s['sample_id'], split=s['split'],
                input_sha256=s['input_sha256'], input_shape=s['input_shape'], **timing,
                metric_conversion=conversion, calibration=s['calibration'],
                calibration_warning=s['calibration_warning'], canonical_units='official raw canonical',
                metric_units='metres' if metric is not None else None,
                run_config=config, output_sha256={n: digest_file(folder/n) for n in outputs}))
            records.append(dict(sample_id=s['sample_id'], status='success', **timing))
            print('OK', s['sample_id'], timing['seconds'], flush=True)
        except Exception as exc:
            failure = dict(sample_id=s['sample_id'], error=f'{type(exc).__name__}: {exc}')
            failures.append(failure)
            records.append(dict(status='failed', **failure))
            print('FAILED', failure, flush=True)
            # A model initialization failure is common to all images; stop retrying.
            if predictor is None and not (folder / 'prediction.json').exists():
                break
        finally:
            write_json(output / 'inference_failures.json', failures)
            write_json(output / 'inference_summary.json', dict(selected=len(selected), records=records,
                failed=len(failures), pending=len(selected)-len(records)))
    evaluation_failures, figure_failures = [], []
    if not args.skip_evaluation:
        evaluation_failures = evaluate_saved(output, seed=args.seed, pair_draws=args.pair_draws,
            gt_gap_m=args.gt_gap_m, ring_radius=args.ring_radius, boundary_margin=args.boundary_margin,
            compression_ratio=args.compression_ratio)
    if not args.skip_visualization:
        figure_failures = visualize_saved(output, depth_max=args.depth_max, error_max=args.error_max)
    meta = read_json(metadata_path)
    failed = failures or evaluation_failures or figure_failures
    meta.update(status='completed_with_failures' if failed else 'completed',
                evaluation_failures=len(evaluation_failures), visualization_failures=len(figure_failures))
    write_json(metadata_path, meta)
    return 1 if failed else 0


if __name__ == '__main__':
    a = parser().parse_args()
    if a.print_config:
        print_config(a)
    else:
        raise SystemExit(run(a))
