"""Manifest-bound opaque-only inference; transparent outputs are read-only."""
import copy
import tempfile
from pathlib import Path
import numpy as np
from src.common import read_json, write_json, write_npy, digest_file, protect_dataset
from src.datasets.cleargrasp import read_rgb, path_for, calibration, annotations
from src.datasets.cleargrasp_paths import full_sample_id, resolve_prediction_dir, load_result_manifest
from src.evaluation.cleargrasp import saved_prediction
from src.inference.da3_zero_shot import (DA3Predictor, metric_depth_from_canonical,
    resize_prediction, MODEL_ID, DA3_COMMIT, METRIC_PROTOCOL)


def identity(sample):
    key = full_sample_id(sample)
    if sample['split'] != 'real-test':
        raise ValueError('Paired experiment accepts existing real-test samples only')
    return key


def contained(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if root not in path.resolve().parents:
        raise ValueError('Path escapes root')
    # Refuse links in outputs and source result paths, including intermediate links.
    current = path
    while current != root:
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('Linked result paths are unsupported')
        current = current.parent
    return path


def opaque_folder(output, sample):
    return contained(output, 'predictions/'+identity(sample))


def source_folder(output, sample):
    identity(sample)
    return resolve_prediction_dir(output, sample)


def load_transparent(output, sample):
    depth, _, meta = saved_prediction(output, sample, require_metric=True)
    return depth, meta


def load_metric(folder, sample, opaque=False):
    """Validate bytes and calibrated metric provenance without reconverting depth."""
    prefix = 'opaque_' if opaque else ''
    meta = read_json(contained(folder, prefix+'prediction.json'))
    if identity(meta) != identity(sample):
        raise ValueError('Prediction identity mismatch')
    hashes = meta.get('output_sha256', {})
    required = prefix+'metric_depth.npy'
    if required not in hashes:
        raise ValueError('Missing metric prediction checksum')
    for name, sha in hashes.items():
        if name not in (prefix+'metric_depth.npy', prefix+'canonical_depth.npy'):
            raise ValueError('Unexpected checksum filename')
        if digest_file(contained(folder, name)) != sha:
            raise ValueError('Prediction checksum mismatch')
    depth = np.load(contained(folder, required), allow_pickle=False)
    if depth.ndim != 2 or not (np.isfinite(depth) & (depth > 0)).any():
        raise ValueError('Invalid metric depth')
    c, cal = meta.get('metric_conversion', {}), meta.get('calibration', {})
    if (not c or c.get('protocol') != METRIC_PROTOCOL or c.get('gt_used') is not False or
            c.get('applied_exactly_once') is not True or meta.get('metric_units') != 'metres'):
        raise ValueError('Untrusted metric conversion')
    if (meta.get('input_shape') != cal.get('image_shape') or
            meta.get('input_shape') != c.get('input_shape') or list(depth.shape) != c.get('native_shape')):
        raise ValueError('Inconsistent calibration/output dimensions')
    _, expected = metric_depth_from_canonical(np.ones(depth.shape), cal['K'], cal['image_shape'])
    for key in ('multiplier', 'focal_native_px', 'input_K', 'native_K'):
        if not np.allclose(c[key], expected[key], rtol=1e-7, atol=1e-8):
            raise ValueError('Inconsistent focal conversion')
    return depth, meta


def validate_paths(a):
    data, source, output = [Path(x).resolve() for x in (a.data_root, a.transparent_output, a.output_dir)]
    protect_dataset(data, output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Paired output and transparent source must be disjoint')
    if output == data or output in data.parents:
        raise ValueError('Output must not contain the dataset')
    checkpoint = a.checkpoint.resolve()
    if output == checkpoint or output in checkpoint.parents or checkpoint in output.parents:
        raise ValueError('Output and checkpoint must be disjoint')


def prepare(a):
    validate_paths(a)
    source = Path(a.transparent_output)
    manifest = load_result_manifest(source, expected_samples=a.expected_samples)
    samples = manifest['samples']
    if len(samples) != a.expected_samples or len({identity(s) for s in samples}) != len(samples):
        raise ValueError(f'Original manifest must contain exactly {a.expected_samples} unique samples')
    run = read_json(source/'run_metadata.json')['run_config']
    if [identity(s) for s in run.get('selection', [])] != [identity(s) for s in samples]:
        raise ValueError('Source manifest selection differs from transparent run metadata')
    hashes = {n: digest_file(a.checkpoint/n) for n in ('config.json', 'model.safetensors')}
    required = dict(model_id=MODEL_ID, da3_commit=DA3_COMMIT, checkpoint_hashes=hashes,
        process_res=a.process_res, process_res_method='upper_bound_resize', precision=a.precision,
        device=a.device, offline=True, gt_input=False)
    for key, value in required.items():
        if run.get(key) != value:
            raise ValueError(f'Transparent inference protocol mismatch: {key}')
    reviews = read_json(a.registration_review) if a.registration_review else {}
    if not isinstance(reviews, dict) or set(reviews)-{identity(s) for s in samples}:
        raise ValueError('Registration review must map original full sample IDs only')
    records = []
    for original in samples:
        s = copy.deepcopy(original)
        rec = dict(sample=s, pair_id=identity(s), status='excluded', camera=identity(s).split('/')[1],
            scene=None, sample_group=None, registration=dict(verified=False, reason='No registration review'))
        try:
            split, camera, number = identity(s).split('/')
            expected_files = dict(rgb=f'{split}/{camera}/{number}-transparent-rgb-img.jpg',
                gt=f'{split}/{camera}/{number}-opaque-depth-img.exr',
                mask=f'{split}/{camera}/{number}-mask.png', intrinsics=f'{split}/{camera}/camera_intrinsics.yaml')
            if any(s['files'].get(k) != v for k, v in expected_files.items()):
                raise ValueError('Manifest files disagree with split/camera/sample or opaque GT convention')
            rgb = read_rgb(path_for(a.data_root, s, 'rgb'))
            shape = rgb.shape[:2]
            folder = source_folder(source, s)
            _, meta = load_transparent(source, s)
            if meta['run_config'] != run or meta['input_sha256'] != digest_file(path_for(a.data_root, s, 'rgb')):
                raise ValueError('Transparent RGB/run provenance differs')
            cal = calibration(a.data_root, s, shape)
            if cal != meta['calibration']:
                raise ValueError('Current and saved calibration differ')
            opaque = f'{split}/{camera}/{number}-opaque-rgb-img.jpg'
            opaque_path = contained(a.data_root, opaque)
            if not opaque_path.is_file():
                raise FileNotFoundError(f'missing pair: {opaque}')
            if read_rgb(opaque_path).shape != rgb.shape:
                raise ValueError('Opaque/transparent resolution mismatch; no automatic crop/warp')
            gt, mask, annotation_info = annotations(a.data_root, s, shape)
            if not (np.isfinite(gt) & (gt > 0)).any():
                raise ValueError('No valid GT')
            review = reviews.get(identity(s), {})
            verified = review.get('verified') is True
            if verified:
                for key, value in dict(transparent_sha256=meta['input_sha256'],
                    opaque_sha256=digest_file(opaque_path), gt_sha256=annotation_info['gt_sha256'],
                    mask_sha256=annotation_info['mask_sha256']).items():
                    if review.get(key) != value:
                        raise ValueError(f'Registration review checksum missing/mismatch: {key}')
                if not review.get('reason') or not review.get('reviewer'):
                    raise ValueError('Registration approval requires reviewer and evidence/reason')
            scene = review.get('scene')
            if scene is not None and (not isinstance(scene, str) or not scene.strip()):
                raise ValueError('Scene must be a nonempty verified group string')
            if scene and (not review.get('reviewer') or not review.get('reason')):
                raise ValueError('Scene groups require reviewer and source evidence/reason')
            rec.update(status='ready', opaque_rgb=opaque, input_shape=list(shape), calibration=cal,
                opaque_sha256=digest_file(opaque_path), transparent_folder=str(folder.resolve()),
                transparent_prediction_sha256=digest_file(folder/'prediction.json'),
                transparent_metric_sha256=meta['output_sha256']['metric_depth.npy'],
                annotations=annotation_info, registration=dict(review, verified=verified,
                    reason=review.get('reason', 'Unverified paired capture; pixel differences skipped')),
                scene=scene, sample_group=f'{camera}/{scene}' if scene else None)
        except Exception as exc:
            rec['reason'] = f'{type(exc).__name__}: {exc}'
        records.append(rec)
    return dict(schema_version=1, source_manifest=str((source/'sample_manifest.json').resolve()),
        source_manifest_sha256=digest_file(source/'sample_manifest.json'),
        source_metadata_sha256=digest_file(source/'run_metadata.json'),
        data_root=str(a.data_root.resolve()), transparent_output=str(source.resolve()),
        inference_protocol=required, records=records)


def read_pair(a, rec):
    """Recheck sources on CPU analysis too; do not trust mutable path references."""
    s = rec['sample']
    folder = source_folder(Path(a.transparent_output), s)
    if digest_file(folder/'prediction.json') != rec['transparent_prediction_sha256']:
        raise ValueError('Transparent metadata changed since preflight')
    t, tm = load_transparent(Path(a.transparent_output), s)
    o, om = load_metric(opaque_folder(a.output_dir, s), s, opaque=True)
    opaque_path = contained(a.data_root, rec['opaque_rgb'])
    if digest_file(path_for(a.data_root, s, 'rgb')) != tm['input_sha256']:
        raise ValueError('Transparent RGB changed')
    if digest_file(opaque_path) != rec['opaque_sha256'] or om['input_sha256'] != rec['opaque_sha256']:
        raise ValueError('Opaque RGB changed')
    if om.get('pair_record') != rec or om.get('inference_protocol') != read_json(
            a.output_dir/'paired_manifest.json')['inference_protocol']:
        raise ValueError('Opaque provenance differs')
    if t.shape != o.shape or om['calibration'] != tm['calibration']:
        raise ValueError('Native grids or calibration differ')
    shape = rec['input_shape']
    gt, mask, info = annotations(a.data_root, s, shape)
    if info != rec['annotations'] or calibration(a.data_root, s, shape) != rec['calibration']:
        raise ValueError('Annotations/calibration changed since preflight')
    return resize_prediction(t, shape), resize_prediction(o, shape), gt, mask


def run(a):
    manifest = prepare(a)
    ready = [r for r in manifest['records'] if r['status'] == 'ready']
    excluded = [r for r in manifest['records'] if r['status'] != 'ready']
    print(f'Original samples: {len(manifest["records"])}; ready: {len(ready)}; excluded: {len(excluded)}')
    for rec in excluded:
        print(rec['pair_id'], rec['reason'])
    if a.dry_run:
        print(__import__('json').dumps(manifest, indent=2))
        return int(bool(excluded))
    out = a.output_dir
    if out.exists() and any(out.iterdir()):
        if not a.resume:
            raise FileExistsError('Output exists; use --resume for verified continuation')
        if read_json(out/'paired_manifest.json') != manifest:
            raise ValueError('Resume manifest/provenance differs; use a new output directory')
    else:
        write_json(out/'paired_manifest.json', manifest)
    config = manifest['inference_protocol']
    write_json(out/'run_metadata.json', dict(status='running', run_config=config,
        source_manifest=manifest['source_manifest'], transparent_inference=False))
    predictor, failures, records = None, list(excluded), []
    for rec in ready:
        folder = opaque_folder(out, rec['sample'])
        try:
            if (folder/'opaque_prediction.json').exists():
                read_pair(a, rec)
                records.append(dict(pair_id=rec['pair_id'], status='resumed'))
                continue
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError('Partial opaque files exist; preserve them and select a new output directory')
            if predictor is None:
                predictor = DA3Predictor(checkpoint=a.checkpoint, device=a.device, precision=a.precision)
            rgb = read_rgb(contained(a.data_root, rec['opaque_rgb']))
            canonical, timing = predictor.predict_canonical(rgb, a.process_res, 'upper_bound_resize')
            metric, conversion = metric_depth_from_canonical(canonical, rec['calibration']['K'], rgb.shape[:2])
            t, _ = load_transparent(Path(a.transparent_output), rec['sample'])
            if t.shape != metric.shape:
                raise ValueError('Opaque native output grid differs from transparent grid')
            staging_root = contained(out, '.staging')
            staging_root.mkdir(parents=True, exist_ok=True)
            staged = Path(tempfile.mkdtemp(prefix='opaque-', dir=staging_root))
            write_npy(staged/'opaque_canonical_depth.npy', canonical)
            write_npy(staged/'opaque_metric_depth.npy', metric)
            write_json(staged/'opaque_prediction.json', dict(sample_id=rec['sample']['sample_id'],
                split=rec['sample']['split'], input_shape=rec['input_shape'], input_sha256=rec['opaque_sha256'],
                calibration=rec['calibration'], metric_conversion=conversion, metric_units='metres',
                pair_record=rec, inference_protocol=config, predictor=predictor.metadata, **timing,
                output_sha256={n: digest_file(staged/n) for n in
                    ('opaque_canonical_depth.npy', 'opaque_metric_depth.npy')}))
            folder.parent.mkdir(parents=True, exist_ok=True)
            staged.rename(folder)  # Complete folder becomes visible atomically; no file overwrite.
            records.append(dict(pair_id=rec['pair_id'], status='success', **timing))
        except Exception as exc:
            failures.append(dict(pair_id=rec['pair_id'], reason=f'{type(exc).__name__}: {exc}'))
            if predictor is None:
                break  # Initialization failures are common to all remaining samples.
        finally:
            write_json(out/'inference_failures.json', failures)
            write_json(out/'inference_summary.json', dict(selected=len(manifest['records']),
                successful=len(records), excluded=len(excluded), records=records,
                failed=len(failures)-len(excluded), pending=len(ready)-len(records)-(len(failures)-len(excluded))))
    # Also produce accounting when every selected sample was excluded.
    write_json(out/'inference_failures.json', failures)
    if not ready:
        write_json(out/'inference_summary.json', dict(selected=len(manifest['records']), successful=0,
            excluded=len(excluded), records=[], failed=0, pending=0))
    write_json(out/'run_metadata.json', dict(status='completed_with_failures' if failures else 'completed',
        run_config=config, transparent_inference=False, source_manifest=manifest['source_manifest'],
        paired_manifest_sha256=digest_file(out/'paired_manifest.json'),
        predictor=predictor.metadata if predictor else None))
    return int(bool(failures))
