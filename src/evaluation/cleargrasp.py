"""Evaluate saved predictions with explicit raw versus oracle protocols."""
import hashlib
from pathlib import Path
import numpy as np
from src.common import read_json, write_json, write_csv, sample_output, digest_file, protect_dataset
from src.datasets.cleargrasp import annotations, path_for, read_rgb
from src.inference.da3_zero_shot import resize_prediction
from src.evaluation.metrics import MetricAccumulator
from src.evaluation.relative_geometry import relative_geometry, REGIONS, ORACLE_LABEL


def saved_prediction(output, sample):
    folder = sample_output(output, sample['split'], sample['sample_id'])
    meta = read_json(folder / 'prediction.json')
    if meta['sample_id'] != sample['sample_id'] or meta['split'] != sample['split']:
        raise ValueError('Prediction identity mismatch')
    for name, sha in meta['output_sha256'].items():
        if name not in ('canonical_depth.npy', 'metric_depth.npy') or digest_file(folder / name) != sha:
            raise ValueError('Saved prediction integrity failure')
    metric = meta['metric_conversion'] is not None
    name = 'metric_depth.npy' if metric else 'canonical_depth.npy'
    a = np.load(folder / name, allow_pickle=False)
    if a.ndim != 2 or not (np.isfinite(a) & (a > 0)).any():
        raise ValueError('Invalid prediction array')
    return a, metric, meta


def sample_seed(seed, sample):
    key = f"{seed}/{sample['split']}/{sample['sample_id']}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')


def evaluate_saved(output, data_root=None, seed=42, **protocol):
    output = Path(output)
    manifest = read_json(output / 'sample_manifest.json')
    root = data_root or manifest['data_root']
    protect_dataset(root, output)
    accum = {r: MetricAccumulator() for r in REGIONS}
    rows, raw_rows, diagnostic_rows, failures = [], [], [], []
    for s in manifest['samples']:
        try:
            shape = read_rgb(path_for(root, s, 'rgb')).shape[:2]
            a, metric, meta = saved_prediction(output, s)
            if digest_file(path_for(root, s, 'rgb')) != meta['input_sha256']:
                raise ValueError('RGB changed since inference')
            gt, mask, annotation_info = annotations(root, s, shape)
            pred = resize_prediction(a, shape)
            masks = dict(full_image=np.ones(shape, bool), transparent=mask, background=~mask)
            for region in REGIONS:
                raw = MetricAccumulator().update(pred, gt, masks[region]).result() if metric else dict(
                    status='unavailable', reason='No trusted camera calibration; canonical depth has no metric units')
                if metric:
                    accum[region].update(pred, gt, masks[region])
                raw_rows.append(dict(sample_id=s['sample_id'], split=s['split'], region=region, **raw))
            result = relative_geometry(pred, gt, mask, seed=sample_seed(seed, s), **protocol)
            result.update(sample_id=s['sample_id'], split=s['split'], prediction_kind='metric' if metric else 'canonical',
                          annotations=annotation_info)
            rows.append(result)
            for region, d in result['diagnostics']['regions'].items():
                diagnostic_rows.append(dict(sample_id=s['sample_id'], split=s['split'], region=region,
                    oracle_scale=result['diagnostics']['oracle_scale'],
                    background_oracle_scale=result['diagnostics']['background_oracle_scale'], **d))
        except Exception as exc:
            failures.append(dict(sample_id=s['sample_id'], split=s['split'], error=f'{type(exc).__name__}: {exc}'))
    def pair_summary(kind):
        entries = [r['ordinal'][kind] for r in rows]
        n = sum(e['valid_pairs'] for e in entries)
        correct = sum(e['correct_pairs'] for e in entries)
        return dict(status='available' if n else 'unavailable', valid_pairs=n, correct_pairs=correct,
                    accuracy=correct/n if n else None)
    ordinal = {kind: pair_summary(kind) for kind in
               ('full_image', 'transparent_transparent', 'transparent_background', 'background_background')}
    n = sum(r['penetration']['valid_pairs'] for r in rows)
    penetration = dict(status='available' if n else 'unavailable', valid_pairs=n)
    for key in ('error', 'reversed', 'compressed'):
        total = sum(r['penetration'][key+'_pairs'] for r in rows)
        penetration[key+'_pairs'] = total
        penetration[key+'_rate'] = total/n if n else None
    regions = {}
    for region in REGIONS:
        ds = [r['diagnostics']['regions'][region] for r in rows
              if r['diagnostics']['regions'][region]['valid_pixels']]
        regions[region] = dict(samples=len(ds), **{key: float(np.mean([d[key] for d in ds
            if d[key] is not None])) if any(d[key] is not None for d in ds) else None for key in
            ('oracle_absrel', 'background_oracle_absrel', 'silog', 'median_log_scale_bias', 'log_residual_mad')})
    metrics = output / 'metrics'
    raw = dict(protocol='unscaled metric metres; finite positive GT AND prediction; no GT alignment',
               aggregation='global pixel weighted', regions={r: acc.result() for r, acc in accum.items()})
    summary = dict(label=ORACLE_LABEL, protocol=dict(seed=seed, **protocol),
                   aggregation='diagnostics: equal image mean; ordinal/penetration: pooled sampled pair counts',
                   regions=regions, ordinal=ordinal, penetration=penetration,
                   evaluated_samples=len(rows), failed_samples=len(failures),
                   splits=sorted({s['split'] for s in manifest['samples']}),
                   benchmark='project zero-shot protocol, not official ClearGrasp benchmark')
    write_json(metrics / 'raw_metric_summary.json', raw)
    write_json(metrics / 'relative_geometry_summary.json', summary)
    write_json(metrics / 'relative_geometry_per_image.json', rows)
    write_json(metrics / 'evaluation_failures.json', failures)
    write_csv(metrics / 'raw_metric_per_image.csv', raw_rows,
              ['sample_id', 'split', 'region', 'status', 'reason', 'valid_pixels', 'samples', 'AbsRel', 'RMSE', 'delta1'])
    write_csv(metrics / 'scale_diagnostics.csv', diagnostic_rows,
              ['sample_id', 'split', 'region', 'oracle_scale', 'background_oracle_scale', 'status',
               'valid_pixels', 'oracle_absrel', 'background_oracle_absrel', 'median_log_scale_bias', 'silog', 'log_residual_mad'])
    flat = []
    for r in rows:
        for name, o in r['ordinal'].items():
            flat.append(dict(sample_id=r['sample_id'], split=r['split'], comparison=name, **o))
    write_csv(metrics / 'relative_geometry_per_image.csv', flat,
              ['sample_id', 'split', 'comparison', 'status', 'valid_pairs', 'correct_pairs', 'accuracy'])
    table = ['| Region | Raw AbsRel | Oracle Aligned AbsRel | SILog | Ordinal Accuracy |',
             '|---|---|---|---|---|']
    for region, comparison in zip(REGIONS, ('full_image', 'transparent_transparent', 'background_background')):
        values = (raw['regions'][region]['AbsRel'], regions[region]['oracle_absrel'],
                  regions[region]['silog'], ordinal[comparison]['accuracy'])
        table.append('| ' + region + ' | ' + ' | '.join('unavailable' if x is None else f'{x:.6f}' for x in values) + ' |')
    table.extend(['', ORACLE_LABEL, '', 'Ordinal regional columns: full/full, T/T and B/B respectively.',
                  f"Transparent/background: {ordinal['transparent_background']}", f'Penetration: {penetration}',
                  '', summary['aggregation']])
    (metrics / 'summary.md').write_text('\n'.join(table), encoding='utf-8')
    return failures
