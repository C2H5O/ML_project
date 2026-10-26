"""CPU-only analysis of an immutable saved manifest and metric predictions.

Never calls a predictor, selects new samples, changes predictions or writes metrics/.
"""
from pathlib import Path
import uuid
import numpy as np

from src.common import read_json, write_json, write_csv, digest_file, protect_dataset
from src.datasets.cleargrasp import annotations, read_rgb, path_for, calibration
from src.datasets.cleargrasp_paths import (load_result_manifest, full_sample_id,
    resolve_prediction_dir, safe_result_path)
from src.evaluation.cleargrasp import saved_prediction, sample_seed
from src.evaluation.metrics import MetricAccumulator, valid_pixels
from src.evaluation.relative_geometry import relative_geometry, REGIONS, ORACLE_LABEL
from src.evaluation.cleargrasp_boundary import boundary_geometry
from src.inference.da3_zero_shot import resize_prediction


def flatten(record, prefix=''):
    result = {}
    for key, value in record.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif not isinstance(value, (list, tuple, np.ndarray)):
            result[name] = value
    return result


def save_table(path, rows):
    rows = [flatten(r) for r in rows]
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['sample_id']
    write_csv(path, rows, fields)


def compare_raw_baseline(recomputed, baseline, rtol=1e-6, atol=1e-8):
    if baseline is None:
        return dict(status='unavailable', reason='No previous raw_metric_summary.json', differences=[])
    if (not isinstance(baseline, dict) or not isinstance(baseline.get('regions'), dict) or
            any(not isinstance(baseline['regions'].get(r), dict) for r in REGIONS)):
        return dict(status='mismatch', reason='Malformed previous raw baseline schema', differences=[],
                    previous_baseline_overwritten=False)
    differences = []
    for region in REGIONS:
        old = baseline.get('regions', {}).get(region, {})
        new = recomputed['regions'][region]
        for key in ('valid_pixels', 'samples', 'AbsRel', 'RMSE', 'delta1', 'status'):
            a, b = old.get(key), new.get(key)
            if key in ('AbsRel', 'RMSE', 'delta1') and a is not None and b is not None:
                equal = isinstance(a, (float, int)) and np.isclose(a, b, rtol=rtol, atol=atol)
            else:
                equal = a == b
            if not equal:
                differences.append(dict(region=region, metric=key, previous=a, recomputed=b))
    if baseline.get('aggregation') != 'global pixel weighted':
        differences.append(dict(metric='aggregation', previous=baseline.get('aggregation'),
                                recomputed='global pixel weighted'))
    return dict(status='match' if not differences else 'mismatch', differences=differences,
                relative_tolerance=rtol, absolute_tolerance=atol, previous_baseline_overwritten=False)


def _ratio_change(raw, diagnostic):
    return (raw-diagnostic)/raw if raw is not None and raw > 0 and diagnostic is not None else None


def scale_distributions(rows):
    """Image-weighted descriptive summaries, distinct from pixel-weighted metrics."""
    result = {}
    for key in ('median_pred_gt', 'oracle_scale', 'background_oracle_scale',
                'transparent_minus_background', 'transparent_relative_increase'):
        values = [r['scale'][key] for r in rows if r['scale'][key] is not None]
        result[key] = dict(images=len(values), median=float(np.median(values)) if values else None,
                          minimum=min(values) if values else None, maximum=max(values) if values else None)
    return dict(aggregation='per-image descriptive statistics, not pooled pixel metrics', values=result)


def scale_row(sample_id, raw, diagnostics):
    regions = diagnostics['regions']
    t = regions['transparent']['background_oracle_absrel']
    b = regions['background']['background_oracle_absrel']
    difference = t-b if t is not None and b is not None else None
    full = regions['full_image']
    return dict(sample_id=sample_id, label=ORACLE_LABEL,
        median_pred_gt=full['median_pred_gt'], oracle_scale=diagnostics['oracle_scale'],
        background_oracle_scale=diagnostics['background_oracle_scale'],
        raw_absrel=raw['full_image']['AbsRel'], oracle_absrel=full['oracle_absrel'],
        oracle_absrel_reduction_fraction=_ratio_change(raw['full_image']['AbsRel'], full['oracle_absrel']),
        median_log_scale_bias=full['median_log_scale_bias'], silog=full['silog'],
        log_residual_mad=full['log_residual_mad'],
        transparent_background_aligned_absrel=t, background_background_aligned_absrel=b,
        transparent_minus_background=difference,
        transparent_relative_increase=difference/b if difference is not None and b > 0 else None,
        relative_increase_note='unavailable for zero or missing background residual; no infinity is reported')


def pooled_pairs(entries, penetration=False):
    usable = [e for e in entries if e['status'] == 'available']
    n = sum(e['valid_pairs'] for e in usable)
    result = dict(status='available' if n else 'unavailable', supported_images=len(usable),
        unavailable_images=len(entries)-len(usable), valid_pairs=n,
        all_images_valid_pairs=sum(e['valid_pairs'] for e in entries),
        unique_pairs=sum(e['unique_pairs'] for e in usable),
        candidate_draws=sum(e.get('candidate_draws', 0) for e in entries),
        aggregation='pooled sampled pairs from per-image support-qualified results only')
    result['all_images_accepted_draw_fraction'] = (result['all_images_valid_pairs']/result['candidate_draws']
                                                  if result['candidate_draws'] else None)
    if penetration:
        for count, rate in [('reversed_pairs', 'reversal_rate'), ('compressed_pairs', 'compression_rate'),
                            ('error_pairs', 'error_rate')]:
            result[count] = sum(e[count] for e in usable)
            result[rate] = result[count]/n if n else None
    else:
        result['correct_pairs'] = sum(e['correct_pairs'] for e in usable)
        result['accuracy'] = result['correct_pairs']/n if n else None
    return result


def hardest_samples(rows, count=3):
    metrics = {
        'highest_raw_absrel': (lambda r: r['raw']['full_image']['AbsRel'], True),
        'highest_background_aligned_transparent_absrel': (
            lambda r: r['scale']['transparent_background_aligned_absrel'], True),
        'lowest_transparent_background_ordinal_accuracy': (
            lambda r: r['geometry']['ordinal']['transparent_background']['accuracy'], False),
        'highest_local_penetration_error_rate': (lambda r: r['geometry']['penetration']['error_rate'], True)}
    result = {}
    for name, (key, reverse) in metrics.items():
        supported = [r for r in rows if key(r) is not None]
        ordered = sorted(supported, key=lambda r: (-key(r) if reverse else key(r), r['sample_id']))[:count]
        result[name] = [dict(sample_id=r['sample_id'], value=key(r), figure=r['figure']) for r in ordered]
    return result


def _resolve_data_root(configured, manifest):
    root = Path(configured).resolve()
    probe = manifest['samples'][0]['files']['rgb']
    if path_for(root, manifest['samples'][0], 'rgb').is_file():
        return root
    # Resolve a single wrapper without rediscovering RGBs or selecting any samples.
    candidates = [p for p in root.iterdir() if p.is_dir() and (p/probe).is_file()] if root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    return root  # Individual missing-file errors retain manifest identities.


def _load_validated_sample(output, root, sample):
    native, _, meta = saved_prediction(output, sample, require_metric=True)
    rgb_path = path_for(root, sample, 'rgb')
    if digest_file(rgb_path) != meta.get('input_sha256'):
        raise ValueError('Input RGB checksum differs from the inference metadata')
    rgb = read_rgb(rgb_path)
    if list(rgb.shape[:2]) != meta.get('input_shape'):
        raise ValueError('RGB dimensions differ from the saved input grid')
    current = calibration(root, sample, rgb.shape[:2])
    original = meta['calibration']
    if (current is None or current['sha256'] != original['sha256'] or
            current['source'] != original['source'] or current['camera'] != original.get('camera') or
            not np.allclose(current['K'], original['K'], rtol=0, atol=1e-8)):
        raise ValueError('Current camera calibration does not match the recorded inference calibration')
    gt, mask, annotation_info = annotations(root, sample, rgb.shape[:2])
    pred = resize_prediction(native, gt.shape)
    if not valid_pixels(pred, gt).any():
        raise ValueError('No valid GT/prediction intersection for this saved sample')
    folder = resolve_prediction_dir(output, sample)
    provenance = dict(prediction_path=folder.relative_to(Path(output).absolute()).as_posix(),
        prediction_json_sha256=digest_file(folder/'prediction.json'),
        output_sha256=meta['output_sha256'], input_sha256=meta['input_sha256'],
        calibration_source=original['source'], calibration_sha256=original['sha256'],
        metric_units=meta['metric_units'], metric_conversion=meta['metric_conversion'], annotations=annotation_info)
    return rgb, pred, gt, mask, provenance


def analyze_saved_geometry(output, data_root, expected_samples=20, split='real-test', seed=42,
        pair_draws=20000, gt_gap_m=.02, ring_radius=30, boundary_margin=2, compression_ratio=.25,
        min_pairs=50, min_unique_pixels=10, boundary_width=4, min_region_pixels=10,
        depth_max=2., error_max=.5, baseline_rtol=1e-6, baseline_atol=1e-8):
    """Read all saved samples; publish only a separate analysis report directory."""
    output = Path(output).absolute()
    protect_dataset(data_root, output)
    manifest = load_result_manifest(output, expected_samples)
    if {s['split'] for s in manifest['samples']} != {split}:
        raise ValueError('Configured split differs from the saved manifest; no resampling is performed')
    root = _resolve_data_root(data_root, manifest)
    protect_dataset(root, output)
    if (min_pairs < 1 or min_unique_pixels < 1 or min_region_pixels < 1 or pair_draws < 1 or
            not 0 <= boundary_margin < boundary_width <= ring_radius or gt_gap_m <= 0 or
            not 0 < compression_ratio < 1 or min(depth_max, error_max) <= 0 or
            min(baseline_rtol, baseline_atol) < 0):
        raise ValueError('Invalid geometry analysis parameters')
    if (output/'run_metadata.json').exists():
        original_selection = read_json(output/'run_metadata.json').get('run_config', {}).get('selection')
        if original_selection is not None and sorted(full_sample_id(s) for s in original_selection) != sorted(
                full_sample_id(s) for s in manifest['samples']):
            raise ValueError('Saved run selection and manifest disagree')
    target = safe_result_path(output, 'geometry_analysis')
    if target.exists():
        raise FileExistsError(f'Analysis directory already exists: {target}. Archive/rename that report directory '
                              'before rerunning; do not change the saved prediction directory or manifest.')
    stage = safe_result_path(output, f'.geometry_analysis-{uuid.uuid4().hex}.partial')
    stage.mkdir()
    manifest_sha = digest_file(output/'sample_manifest.json')
    baseline_path = safe_result_path(output, 'metrics', 'raw_metric_summary.json')
    baseline_sha = digest_file(baseline_path) if baseline_path.is_file() else None
    baseline_error, baseline = None, None
    if baseline_sha:
        try:
            baseline = read_json(baseline_path)
        except Exception as exc:
            baseline_error = str(exc)
    protocol = dict(seed=seed, pair_draws=pair_draws, gt_gap_m=gt_gap_m, ring_radius=ring_radius,
        boundary_margin=boundary_margin, compression_ratio=compression_ratio,
        min_unique_pairs=min_pairs, min_unique_pixels=min_unique_pixels,
        boundary_width=boundary_width, min_region_pixels=min_region_pixels,
        raw_gt_alignment=False, diagnostic_label=ORACLE_LABEL,
        local_pair_limit='Nearby background is not measured occluded background on the same camera ray',
        expected_samples=expected_samples, split=split, no_model_called=True, saved_manifest_used=True)
    validation = []
    print(f'Validating all {len(manifest["samples"])} saved metric predictions on CPU', flush=True)
    for sample in manifest['samples']:
        try:
            *_, provenance = _load_validated_sample(output, root, sample)
            validation.append(dict(sample_id=full_sample_id(sample), status='verified', provenance=provenance))
        except Exception as exc:
            validation.append(dict(sample_id=full_sample_id(sample), status='failed', error=f'{type(exc).__name__}: {exc}'))
    write_json(stage/'validation.json', validation)
    failures = [v for v in validation if v['status'] == 'failed']
    base = dict(protocol=protocol, manifest_sha256=manifest_sha, previous_baseline_sha256=baseline_sha,
        selected_samples=len(manifest['samples']), verified_samples=len(validation)-len(failures),
        dataset_root=str(root), failed_samples=failures, source_output=str(output),
        prediction_arrays_modified=False, previous_metrics_overwritten=False)
    if failures:
        summary = dict(**base, status='validation_failed', analyzed_samples=0)
        write_json(stage/'summary.json', summary)
        (stage/'report.md').write_text('# ClearGrasp geometry analysis: validation failed\n\n'
            'No aggregate scores or conclusions were produced. Every saved manifest sample was checked.\n\n' +
            '\n'.join(f'- **{r["sample_id"]}**: {r["error"]}' for r in failures), encoding='utf-8')
        if target.exists():
            raise FileExistsError(target)
        stage.rename(target)
        return summary
    from src.visualization.cleargrasp_analysis import render_analysis_sample, render_analysis_charts
    from src.evaluation.cleargrasp_report import write_analysis_report
    raw_acc = {r: MetricAccumulator() for r in REGIONS}
    oracle_acc = {r: MetricAccumulator() for r in REGIONS}
    background_acc = {r: MetricAccumulator() for r in REGIONS}
    rows = []
    try:
        for sample, checked in zip(manifest['samples'], validation):
            rgb, pred, gt, mask, provenance = _load_validated_sample(output, root, sample)
            if provenance != checked['provenance']:
                raise ValueError(f'Sample changed after preflight: {full_sample_id(sample)}')
            masks = dict(full_image=np.ones(gt.shape, bool), transparent=mask, background=~mask)
            raw = {}
            for region, region_mask in masks.items():
                raw[region] = MetricAccumulator().update(pred, gt, region_mask).result()
                raw_acc[region].update(pred, gt, region_mask)
            geometry = relative_geometry(pred, gt, mask, seed=sample_seed(seed, sample),
                pair_draws=pair_draws, gt_gap_m=gt_gap_m, ring_radius=ring_radius,
                boundary_margin=boundary_margin, compression_ratio=compression_ratio,
                min_pairs=min_pairs, min_unique_pixels=min_unique_pixels, return_details=True)
            details = geometry.pop('_details')
            diagnostic = geometry['diagnostics']
            oracle, bg = diagnostic['oracle_scale'], diagnostic['background_oracle_scale']
            for region, region_mask in masks.items():
                if oracle is not None:
                    oracle_acc[region].update(pred*oracle, gt, region_mask)
                if bg is not None:
                    background_acc[region].update(pred*bg, gt, region_mask)
            boundary = boundary_geometry(pred, gt, mask, bg, seed=sample_seed(seed+1, sample),
                pair_draws=pair_draws, gt_gap_m=gt_gap_m, ring_radius=ring_radius,
                boundary_margin=boundary_margin, boundary_width=boundary_width,
                min_pairs=min_pairs, min_unique_pixels=min_unique_pixels,
                min_region_pixels=min_region_pixels, return_details=True)
            boundary_details = boundary.pop('_details')
            sid = full_sample_id(sample)
            figure = render_analysis_sample(stage, sample, rgb, gt, pred, mask, bg, geometry, details,
                                            boundary, boundary_details, depth_max, error_max)
            rows.append(dict(sample_id=sid, raw=raw, geometry=geometry, boundary=boundary,
                scale=scale_row(sid, raw, diagnostic), figure=figure.relative_to(stage).as_posix()))
            print(f'Analyzed {sid}', flush=True)
        raw = dict(protocol='unscaled metric metres; finite positive GT AND prediction; no GT alignment',
                   aggregation='global pixel weighted', regions={r: a.result() for r, a in raw_acc.items()})
        comparison = compare_raw_baseline(raw, baseline, baseline_rtol, baseline_atol)
        if baseline_error:
            comparison = dict(status='mismatch', reason='Could not parse previous baseline', error=baseline_error,
                              differences=[], previous_baseline_overwritten=False)
        diagnostics = dict(label=ORACLE_LABEL, aggregation='global pixel weighted after one per-image diagnostic scale',
            oracle={r: a.result() for r, a in oracle_acc.items()},
            background_aligned={r: a.result() for r, a in background_acc.items()})
        for region in REGIONS:
            diagnostics['oracle'][region]['raw_absrel_reduction_fraction'] = _ratio_change(
                raw['regions'][region]['AbsRel'], diagnostics['oracle'][region]['AbsRel'])
        ordinal = {kind: pooled_pairs([r['geometry']['ordinal'][kind] for r in rows])
                   for kind in rows[0]['geometry']['ordinal']}
        penetration = pooled_pairs([r['geometry']['penetration'] for r in rows], penetration=True)
        boundary_entries = [r['boundary']['contrast'] for r in rows if r['boundary']['contrast']['status'] == 'available']
        bn = sum(e['valid_pairs'] for e in boundary_entries)
        boundary_summary = dict(status='available' if bn else 'unavailable', valid_pairs=bn,
            supported_images=len(boundary_entries), unavailable_images=len(rows)-len(boundary_entries),
            **{key: sum(e[key]*e['valid_pairs'] for e in boundary_entries)/bn if bn else None
               for key in ('normalized_contrast_mae', 'log_contrast_mae', 'reversal_rate', 'tie_rate')})
        summary = dict(**base, status='baseline_mismatch' if comparison['status'] == 'mismatch' else 'complete',
            analyzed_samples=len(rows), raw_metric=raw, baseline_comparison=comparison,
            diagnostic_metric=diagnostics, ordinal=ordinal, penetration=penetration,
            scale_distributions=scale_distributions(rows),
            boundary=boundary_summary, hardest_samples=hardest_samples(rows),
            interpretation='Descriptive diagnostics, not a causal partition of error or a test of statistical significance')
        write_json(stage/'summary.json', summary)
        write_json(stage/'per_image.json', rows)
        write_json(stage/'raw_metric_recomputed.json', raw)
        write_json(stage/'baseline_comparison.json', comparison)
        save_table(stage/'per_image.csv', rows)
        save_table(stage/'scale_analysis.csv', [r['scale'] for r in rows])
        save_table(stage/'ordinal_analysis.csv', [dict(sample_id=r['sample_id'], comparison=name, **o)
                   for r in rows for name, o in r['geometry']['ordinal'].items()])
        save_table(stage/'penetration_analysis.csv', [dict(sample_id=r['sample_id'], **r['geometry']['penetration']) for r in rows])
        save_table(stage/'boundary_analysis.csv', [dict(sample_id=r['sample_id'], **r['boundary']) for r in rows])
        render_analysis_charts(stage, rows, summary)
        write_analysis_report(stage/'report.md', summary, rows)
        # Detect concurrent input edits before publishing; never repair or rewrite them.
        if digest_file(output/'sample_manifest.json') != manifest_sha:
            raise ValueError('Manifest changed during analysis')
        if (digest_file(baseline_path) if baseline_path.is_file() else None) != baseline_sha:
            raise ValueError('Baseline changed during analysis')
        for sample, checked in zip(manifest['samples'], validation):
            _, _, meta = saved_prediction(output, sample, require_metric=True)
            folder = resolve_prediction_dir(output, sample)
            if (meta['output_sha256'] != checked['provenance']['output_sha256'] or
                    digest_file(folder/'prediction.json') != checked['provenance']['prediction_json_sha256']):
                raise ValueError('Saved prediction metadata changed during analysis')
        if target.exists():
            raise FileExistsError(target)
        stage.rename(target)
        return summary
    except Exception as exc:
        write_json(stage/'analysis_failure.json', dict(error=str(exc), status='incomplete',
            recovery='Inputs were not modified. Inspect this partial report directory; fix the failure and rerun.'))
        raise RuntimeError(f'Analysis incomplete; preserved diagnostics at {stage}: {exc}') from exc
