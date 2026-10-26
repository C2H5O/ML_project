"""Shared-support metrics, sample-pair bootstrap, and explicit missing support."""
import numpy as np
from scipy.ndimage import binary_erosion
from src.common import read_json, write_json, write_csv, digest_file
from src.cleargrasp_paired import read_pair, validate_paths, opaque_folder
from src.evaluation.metrics import MetricAccumulator, valid_pixels
from src.evaluation.relative_geometry import oracle_scale
from src.evaluation.paired_geometry import paired_relative_geometry

LABEL = 'GT-dependent diagnostic only — not deployable metric performance.'


def protocol(a):
    return {key: getattr(a, key) for key in ('seed', 'pair_draws', 'gt_gap_m', 'ring_radius',
        'boundary_margin', 'compression_ratio', 'min_pairs', 'min_unique_pixels')}


def differences(t, o, valid, background, verified):
    if not verified:
        return dict(status='skipped', reason='RGB/GT/mask registration not verified')
    delta = np.where(valid, t-o, np.nan)
    log = np.full(t.shape, np.nan)
    log[valid] = np.log(t[valid])-np.log(o[valid])
    bg = log[background & valid]
    return dict(status='available', depth=delta, log_depth=log,
        background_referenced_log=log-np.median(bg) if len(bg) else np.full(t.shape, np.nan))


def compare_pair(t, o, gt, mask, erosion_margin=2, **geometry):
    common = valid_pixels(t, gt) & valid_pixels(o, gt)
    if not common.any():
        raise ValueError('No common finite-positive GT/transparent/opaque support')
    mask = np.asarray(mask, bool)
    if mask.shape != gt.shape:
        raise ValueError('Object mask shape mismatch')
    interiors = (binary_erosion(mask, iterations=erosion_margin),
                 binary_erosion(~mask, iterations=erosion_margin)) if erosion_margin else (mask, ~mask)
    regions = dict(full_image=np.ones(gt.shape, bool), object=mask, background=~mask,
                   object_eroded=interiors[0], background_eroded=interiors[1])
    raw, scale, relative, material = [], [], [], []
    # Both passes preserve the original mask. Eroded pass excludes a boundary band.
    for pass_name, support in (('original', common), ('eroded', common & (interiors[0] | interiors[1]))):
        inputs = [np.where(support, p, np.nan) for p in (t, o)]
        rel = paired_relative_geometry(*inputs, gt, mask, **geometry)
        biases = []
        for name, p, rg in zip(('transparent', 'opaque'), inputs, rel):
            bg_scale = oracle_scale(p, gt, support & ~mask)
            obj_bg = []
            for region, area in regions.items():
                v = support & area
                metrics = MetricAccumulator().update(p, gt, v).result()
                raw.append(dict(input=name, sensitivity=pass_name, region=region, **metrics))
                if v.any():
                    e = np.log(p[v])-np.log(gt[v])
                    ratio, bias = float(np.median(p[v]/gt[v])), float(np.median(e))
                    bg_error = float(np.mean(np.abs(bg_scale*p[v]-gt[v])/gt[v])) if bg_scale else None
                else:
                    ratio, bias, bg_error = None, None, None
                scale.append(dict(input=name, sensitivity=pass_name, region=region, valid_pixels=int(v.sum()),
                    pred_gt_ratio=ratio, log_scale_bias=bias, background_scale=bg_scale,
                    background_aligned_absrel=bg_error, diagnostic_label=LABEL))
                if region in ('object', 'background'):
                    obj_bg.append(bias)
            biases.append(obj_bg[0]-obj_bg[1] if None not in obj_bg else None)
            for region, d in rg['diagnostics']['regions'].items():
                relative.append(dict(input=name, sensitivity=pass_name, category='diagnostic', region=region,
                    oracle_absrel=d['oracle_absrel'], silog=d['silog'], diagnostic_label=LABEL))
            for kind, d in rg['ordinal'].items():
                relative.append(dict(input=name, sensitivity=pass_name, category='ordinal', region=kind, **d))
            relative.append(dict(input=name, sensitivity=pass_name, category='penetration', region='object_background',
                **rg['penetration']))
        material.append(dict(sensitivity=pass_name, transparent_object_minus_background= biases[0],
            opaque_object_minus_background=biases[1],
            delta_material=biases[0]-biases[1] if None not in biases else None,
            object_pixels=int((support & mask).sum()), background_pixels=int((support & ~mask).sum())))
    return dict(raw=raw, scale=scale, relative=relative, material=material,
                common_valid_pixels=int(common.sum()), gt_valid_pixels=int((np.isfinite(gt) & (gt > 0)).sum()))


def bootstrap(values, seed, draws, groups=None):
    values = np.asarray(values, float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    if groups is None:
        means = np.mean(values[rng.integers(0, len(values), (draws, len(values)))], axis=1)
    else:
        pools = [values[np.asarray(groups) == g] for g in sorted(set(groups))]
        if len(pools) < 2:
            return None
        # Equal scene weighting; sample entire clusters, never individual frames.
        scene_means = np.array([x.mean() for x in pools])
        means = scene_means[rng.integers(0, len(pools), (draws, len(pools)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def paired_statistics(values, seed, draws, groups=None, higher_worse=True):
    values = np.asarray(values, float)
    if not len(values):
        return dict(samples=0, mean_difference=None, median_difference=None, bootstrap_95_ci=None)
    score = values if higher_worse else -values
    return dict(samples=len(values), mean_difference=float(values.mean()), median_difference=float(np.median(values)),
        transparent_worse=int((score > 1e-12).sum()), opaque_worse=int((score < -1e-12).sum()),
        ties=int((np.abs(score) <= 1e-12).sum()), bootstrap_95_ci=bootstrap(values, seed, draws),
        scene_bootstrap_95_ci=bootstrap(values, seed, draws, groups) if groups and all(groups) else None,
        scene_mean_difference=(float(np.mean([values[np.asarray(groups) == g].mean() for g in sorted(set(groups))]))
                               if groups and all(groups) else None))


def csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['pair_id', 'status']
    write_csv(path, [{k: row.get(k) for k in fields} for row in rows], fields)


def analysis_identity(a):
    validate_paths(a)
    m = read_json(a.output_dir/'paired_manifest.json')
    if str(a.data_root.resolve()) != m['data_root'] or str(a.transparent_output.resolve()) != m['transparent_output']:
        raise ValueError('Configured source paths differ from saved paired manifest')
    if (digest_file(a.transparent_output/'sample_manifest.json') != m['source_manifest_sha256'] or
            digest_file(a.transparent_output/'run_metadata.json') != m['source_metadata_sha256']):
        raise ValueError('Original source manifest/run metadata changed')
    return m


def output_guard(a, directory, signature):
    """Refuse accidental overwrite; resume verifies the complete artifact set."""
    stamp = directory/'artifact_metadata.json'
    if directory.exists() and any(directory.iterdir()):
        if not a.resume:
            raise FileExistsError(f'{directory} exists; use --resume or choose a new output')
        old = read_json(stamp)
        if old['signature'] != signature:
            raise ValueError('Analysis/visualization protocol differs on resume')
        for name, sha in old['outputs'].items():
            if digest_file(directory/name) != sha:
                raise ValueError('Saved artifact checksum mismatch')
        return False
    directory.mkdir(parents=True, exist_ok=True)
    return True


def finish_artifacts(directory, signature):
    write_json(directory/'artifact_metadata.json', dict(signature=signature,
        outputs={p.relative_to(directory).as_posix(): digest_file(p) for p in directory.rglob('*')
                 if p.is_file() and p.name != 'artifact_metadata.json'}))


def evaluate(a):
    manifest = analysis_identity(a)
    tables = dict(raw=[], scale=[], relative=[], material=[])
    failures, evaluated = [], []
    for rec in manifest['records']:
        if rec['status'] != 'ready':
            failures.append(dict(pair_id=rec['pair_id'], reason=rec['reason']))
            continue
        try:
            arrays = read_pair(a, rec)
            result = compare_pair(*arrays, erosion_margin=a.erosion_margin, **protocol(a))
            evaluated.append(rec)
            for key in tables:
                tables[key].extend(dict(pair_id=rec['pair_id'], camera=rec['camera'], scene=rec['scene'],
                    sample_group=rec['sample_group'], **row) for row in result[key])
        except Exception as exc:
            failures.append(dict(pair_id=rec['pair_id'], reason=f'{type(exc).__name__}: {exc}'))
    signature = dict(manifest_sha256=digest_file(a.output_dir/'paired_manifest.json'),
        protocol=protocol(a), erosion_margin=a.erosion_margin, bootstrap_draws=a.bootstrap_draws,
        predictions={r['pair_id']: digest_file(opaque_folder(
            a.output_dir, r['sample'])/'opaque_prediction.json') for r in evaluated}, failures=failures)
    if a.dry_run:
        print(f'Evaluation preflight: {len(evaluated)} pairs; failures: {failures}')
        return int(bool(failures))
    target = a.output_dir/'metrics'
    if not output_guard(a, target, signature):
        return int(bool(failures))
    # Derive explicit T-O rows and bootstrap every scalar paired metric.
    stats, deltas = {}, []
    for kind in ('raw', 'scale', 'relative'):
        lookup = {}
        for row in tables[kind]:
            key = (row['pair_id'], row['sensitivity'], row['region'], row.get('category', ''))
            lookup.setdefault(key, {})[row['input']] = row
        metric_names = dict(raw=('AbsRel', 'RMSE', 'delta1'), scale=('pred_gt_ratio', 'log_scale_bias',
            'background_aligned_absrel'), relative=('oracle_absrel', 'silog', 'accuracy', 'reversal_rate', 'compression_rate'))[kind]
        for key, pair in lookup.items():
            if len(pair) != 2:
                continue
            t, o = pair['transparent'], pair['opaque']
            for metric in metric_names:
                tv, ov = t.get(metric), o.get(metric)
                if tv is None or ov is None:
                    continue
                name = '/'.join((kind, key[1], key[2], key[3], metric))
                entry = dict(pair_id=key[0], metric=name, transparent=tv, opaque=ov, difference=tv-ov,
                    ratio=tv/ov if metric == 'pred_gt_ratio' and ov else None, sample_group=t['sample_group'])
                deltas.append(entry)
                stats.setdefault(name, []).append(entry)
    for row in tables['material']:
        if row['delta_material'] is not None:
            name = 'material/'+row['sensitivity']
            entry = dict(pair_id=row['pair_id'], metric=name, difference=row['delta_material'],
                sample_group=row['sample_group'])
            deltas.append(entry)
            stats.setdefault(name, []).append(entry)
    summaries = {}
    for name, entries in stats.items():
        signed_bias = name.startswith('material/') or name.endswith(('log_scale_bias', 'pred_gt_ratio'))
        summary = paired_statistics([x['difference'] for x in entries], a.seed, a.bootstrap_draws,
            [x['sample_group'] for x in entries], higher_worse=not name.endswith(('accuracy', 'delta1')))
        if signed_bias:
            summary['positive_differences'] = summary.pop('transparent_worse', 0)
            summary['negative_differences'] = summary.pop('opaque_worse', 0)
            summary['interpretation'] = 'Signed shift; positive does not necessarily mean worse'
        summaries[name] = summary
    names = dict(raw='raw_metric_comparison.csv', scale='scale_comparison.csv',
                 relative='relative_geometry_comparison.csv', material='material_scale_bias.csv')
    for key, rows in tables.items():
        csv(target/names[key], rows)
    csv(target/'paired_differences.csv', deltas)
    write_json(target/'evaluation_failures.json', failures)
    known = {r['sample_group'] for r in evaluated if r['sample_group']}
    summary = dict(selected=len(manifest['records']), evaluated=len(evaluated), excluded_or_failed=len(failures),
        known_scene_groups=len(known), unknown_scene_samples=sum(r['sample_group'] is None for r in evaluated),
        effective_independent_scenes=None if any(r['sample_group'] is None for r in evaluated) else len(known),
        scene_warning='Camera is not a scene. Unknown scene groups prevent independence claims. '
                     'Pair bootstrap is descriptive and may underestimate uncertainty for correlated frames.',
        raw_protocol='same finite-positive GT AND transparent AND opaque intersection; raw metres, no GT alignment',
        aggregation='equal sample-pair weighting; cluster bootstrap equal verified scene weighting',
        diagnostic_label=LABEL, paired_metrics=summaries,
        protocol=protocol(a), erosion_margin=a.erosion_margin, bootstrap_draws=a.bootstrap_draws,
        per_pixel_verified_pairs=sum(r['registration']['verified'] for r in evaluated))
    write_json(target/'paired_summary.json', summary)
    lines = ['# ClearGrasp paired appearance comparison', '',
        f'Selected {summary["selected"]}; evaluated {len(evaluated)}; excluded/failed {len(failures)}.',
        '', summary['raw_protocol'], '', '| Input | Full AbsRel | Object AbsRel | Background AbsRel | Full RMSE | Full delta1 |',
        '|---|---|---|---|---|---|']
    for inp in ('transparent', 'opaque'):
        vals = []
        for region, metric in (('full_image','AbsRel'), ('object','AbsRel'), ('background','AbsRel'),
                               ('full_image','RMSE'), ('full_image','delta1')):
            v = [r[metric] for r in tables['raw'] if r['input'] == inp and r['sensitivity'] == 'original'
                 and r['region'] == region and r[metric] is not None]
            vals.append(f'{np.mean(v):.6f}' if v else 'unavailable')
        lines.append('| '+inp+' | '+' | '.join(vals)+' |')
    lines.extend(['', LABEL, '', summary['scene_warning'],
        f'Known scene groups: {len(known)}; unknown scene samples: {summary["unknown_scene_samples"]}.',
        '', 'Delta material = (median log(T/GT) object - background) - (median log(O/GT) object - background).',
        'Near zero: no additional object/background log scale contrast. Positive: T object is relatively deeper; '
        'negative: relatively shallower, compared with O. Neither sign alone proves worse geometry.',
        'Invariant to independent positive global multipliers on each prediction. Residual capture motion, '
        'lighting and mask misregistration can bias this statistic. Compare original/eroded rows.',
        '', 'All signed differences below are Transparent minus Opaque. Bootstrap resamples sample pairs; '
        'scene CI is available only with complete verified groups and at least two groups.',
        '', '| Metric | N | Mean difference | Median | Pair bootstrap 95% CI |', '|---|---|---|---|---|'])
    for name, st in summaries.items():
        lines.append(f'| {name} | {st["samples"]} | {st["mean_difference"]:.6g} | '
                     f'{st["median_difference"]:.6g} | {st["bootstrap_95_ci"]} |')
    (target/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    finish_artifacts(target, signature)
    return int(bool(failures))
