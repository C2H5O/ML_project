"""Read-only ClearGrasp pairing and annotation protocol. No training dependency.

Sources: Shreeyak/cleargrasp eval_depth_completion/eval_depth_completion.py
and api/utils.py (R channel, metres, finite positive GT, nonzero eval mask).
"""
import importlib.util
import random
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from src.common import digest_file, protect_dataset, write_json

SPLITS = ('real-test', 'real-val', 'synthetic-test', 'synthetic-val')
SOURCE = 'https://github.com/Shreeyak/cleargrasp/blob/master/eval_depth_completion/eval_depth_completion.py'


def read_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))


def exr_capabilities():
    return {name: importlib.util.find_spec(name) is not None
            for name in ('OpenEXR', 'Imath', 'cv2')}


def read_depth(path):
    """Official scalar R channel. Never average RGB channels or auto-scale units."""
    errors = []
    try:
        import OpenEXR
        import Imath
        f = OpenEXR.InputFile(str(path))
        try:
            dw = f.header()['dataWindow']
            a = np.frombuffer(f.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)), np.float32)
            return a.reshape(dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1).copy()
        finally:
            f.close()
    except Exception as exc:
        errors.append(f'OpenEXR: {exc}')
    try:
        # Use only the existing codec configuration; never set environment variables.
        import cv2
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError('EXR decoder returned None (check OpenEXR build support)')
        if a.ndim == 3 and a.shape[2] == 3:
            a = a[..., 2]  # OpenCV BGR -> official R channel
        if a.ndim != 2 or a.dtype.kind != 'f':
            raise ValueError(f'Expected floating scalar EXR, got {a.shape}/{a.dtype}')
        return a.astype(np.float32)
    except Exception as exc:
        errors.append(f'OpenCV: {exc}')
    raise RuntimeError('EXR unavailable; inspect existing OpenEXR/Imath or enabled OpenCV codec; '
                       'no environment-variable changes and no automatic installation. ' + '; '.join(errors))


def read_mask(path, shape=None):
    with Image.open(path) as im:
        a = np.array(im)
    if a.ndim != 2 or (shape is not None and a.shape != tuple(shape)):
        raise ValueError(f'Mask must be scalar and match RGB: {a.shape} vs {shape}')
    values = np.unique(a)
    # Real masks are binary 0/255; synthetic segmentation IDs start at 100.
    if not all(v in (0, 1, 255) or 100 <= v <= 254 for v in values):
        raise ValueError(f'Unrecognized mask encoding: {values.tolist()}')
    return a != 0, dict(shape=list(a.shape), dtype=str(a.dtype), values=values.tolist(),
                        definition='nonzero transparent object; not GT validity', source=SOURCE)


def discover(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not any((root / s).is_dir() for s in SPLITS):
        candidates = [p for p in root.iterdir() if p.is_dir() and any((p / s).is_dir() for s in SPLITS)]
        if len(candidates) != 1:
            raise ValueError('Pass a root containing real-test/real-val/synthetic-test/synthetic-val')
        root = candidates[0]
    samples, splits, orphans = [], {}, []
    for split in SPLITS:
        folder = root / split
        splits[split] = dict(present=folder.is_dir(), samples=0, groups={})
        if not folder.is_dir():
            continue
        real = split.startswith('real')
        suffixes = ({'rgb': '-transparent-rgb-img.jpg', 'gt': '-opaque-depth-img.exr',
                     'mask': '-mask.png', 'sensor_depth': '-transparent-depth-img.exr'} if real else
                    {'rgb': '-rgb.jpg', 'gt': '-depth-rectified.exr', 'mask': '-segmentation-mask.png',
                     'metadata': '-masks.json'})
        groups = {}
        for p in sorted(folder.rglob('*')):
            if not p.is_file():
                continue
            for kind, suffix in suffixes.items():
                if p.name.endswith(suffix):
                    number = p.name[:-len(suffix)]
                    if not number.isdigit():
                        continue  # excludes normal visualizations, etc.
                    group = p.parent if real else p.parent.parent
                    key = (group.relative_to(folder).as_posix(), number)
                    record = groups.setdefault(key, {})
                    if kind in record:
                        raise ValueError(f'Duplicate {kind} for {split}/{key}')
                    record[kind] = p.relative_to(root).as_posix()
        for (group, number), paths in sorted(groups.items()):
            if 'rgb' not in paths:
                orphans.append(dict(split=split, group=group, number=number, files=paths))
                continue
            intrinsic = folder / group / 'camera_intrinsics.yaml'
            if intrinsic.is_file():
                paths['intrinsics'] = intrinsic.relative_to(root).as_posix()
            sample = dict(split=split, sample_id=f'{group}/{number}', group=group, number=number,
                          files=paths, missing=[k for k in ('gt', 'mask') if k not in paths])
            samples.append(sample)
            splits[split]['samples'] += 1
            splits[split]['groups'][group] = splits[split]['groups'].get(group, 0) + 1
    return dict(schema_version=1, dataset='ClearGrasp', data_root=str(root), splits=splits,
                samples=samples, orphan_annotations=orphans, protocol_source=SOURCE)


def path_for(root, sample, kind):
    p = (Path(root) / sample['files'][kind]).resolve()
    if Path(root).resolve() not in p.parents:
        raise ValueError('Manifest path escapes dataset root')
    return p


def calibration(root, sample, shape):
    if 'intrinsics' not in sample['files']:
        return None  # FOV/pose metadata alone is not a verified calibration.
    path = path_for(root, sample, 'intrinsics')
    d = yaml.safe_load(path.read_text(encoding='utf-8'))
    h, w = shape
    if (d['yres'], d['xres']) != (h, w):
        raise ValueError('Calibration dimensions do not match RGB; refusing metric conversion')
    if str(d['camera_name']).lower() != sample['group'].split('/')[-1].lower():
        raise ValueError('Calibration camera does not match sample camera directory')
    K = np.array([[d['fx'], 0, d['cx']], [0, d['fy'], d['cy']], [0, 0, 1]], float)
    if (not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0 or
            not 0 <= K[0, 2] < w or not 0 <= K[1, 2] < h):
        raise ValueError('Invalid pinhole calibration')
    return dict(K=K.tolist(), image_shape=[h, w], source=sample['files']['intrinsics'],
                sha256=digest_file(path), camera=d['camera_name'], content=d)


def annotations(root, sample, shape):
    gt = read_depth(path_for(root, sample, 'gt'))
    mask, info = read_mask(path_for(root, sample, 'mask'), shape)
    if gt.shape != tuple(shape):
        raise ValueError(f'GT/RGB mismatch {gt.shape}/{shape}; annotations are never resized')
    return gt, mask, dict(mask=info, gt_units='metres', gt_channel='R',
                          gt_source=sample['files']['gt'], mask_source=sample['files']['mask'],
                          gt_sha256=digest_file(path_for(root, sample, 'gt')),
                          mask_sha256=digest_file(path_for(root, sample, 'mask')),
                          validity='finite and >0', source=SOURCE,
                          registration='paired capture; pixel dimensions checked, residual motion not corrected')


def select_samples(manifest, split, maximum=20, seed=42, all_samples=False):
    if split not in SPLITS or not manifest['splits'][split]['present']:
        raise ValueError(f'Split unavailable: {split}')
    pool = [s for s in manifest['samples'] if s['split'] == split]
    if not pool:
        raise ValueError(f'No RGB samples: {split}')
    if maximum < 1:
        raise ValueError('max-samples must be positive')
    return pool if all_samples else random.Random(seed).sample(pool, min(maximum, len(pool)))


def inspect_dataset(root, output):
    manifest = discover(root)
    root = Path(manifest['data_root'])
    protect_dataset(root, output)
    report = dict(data_root=str(root), splits=manifest['splits'], exr_capabilities=exr_capabilities(),
                  records=[], missing_files=[], unreadable_files=[], intrinsics={},
                  orphan_annotations=manifest['orphan_annotations'], protocol_source=SOURCE)
    for sample in manifest['samples']:
        rec = dict(split=sample['split'], sample_id=sample['sample_id'], files=sample['files'])
        for kind in sample['missing']:
            report['missing_files'].append(dict(sample_id=sample['sample_id'], split=sample['split'], kind=kind))
        shape = None
        for kind in ('rgb', 'gt', 'sensor_depth', 'mask'):
            if kind not in sample['files']:
                continue
            p = path_for(root, sample, kind)
            try:
                if kind == 'rgb':
                    a = read_rgb(p)
                    shape = a.shape[:2]
                    rec[kind] = dict(shape=list(a.shape), dtype=str(a.dtype), type=p.suffix)
                elif kind == 'mask':
                    _, rec[kind] = read_mask(p, shape)
                    rec[kind]['type'] = p.suffix
                else:
                    a = read_depth(p)
                    rec[kind] = dict(shape=list(a.shape), dtype=str(a.dtype), type=p.suffix,
                                     valid_fraction=float(np.mean(np.isfinite(a) & (a > 0))), units='metres')
                    if shape is not None and a.shape != shape:
                        raise ValueError(f'{kind}/RGB resolution mismatch')
            except Exception as exc:
                rec[kind] = dict(status='unreadable', error=str(exc), type=p.suffix)
                report['unreadable_files'].append(dict(path=str(p), error=str(exc)))
        if 'intrinsics' in sample['files']:
            key = sample['files']['intrinsics']
            try:
                c = calibration(root, sample, shape) if shape else None
                report['intrinsics'][key] = c
            except Exception as exc:
                report['unreadable_files'].append(dict(path=key, error=str(exc)))
        else:
            rec['metric_calibration'] = 'unavailable: no verified camera_intrinsics.yaml; canonical-only inference'
        if 'metadata' in sample['files']:
            try:
                import json
                metadata = json.loads(path_for(root, sample, 'metadata').read_text(encoding='utf-8'))
                camera = metadata.get('camera', {})
                rec['camera_metadata'] = dict(path=sample['files']['metadata'], image=metadata.get('image'),
                    field_of_view=camera.get('field_of_view'), sensor=camera.get('sensor'),
                    focal_length=camera.get('focal_length'),
                    metric_calibration_trusted=False, extrinsics_used=False)
            except Exception as exc:
                report['unreadable_files'].append(dict(path=sample['files']['metadata'], error=str(exc)))
        report['records'].append(rec)
    write_json(Path(output) / 'sample_manifest.json', manifest)
    write_json(Path(output) / 'dataset_inspection.json', report)
    return report
