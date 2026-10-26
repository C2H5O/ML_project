"""Saved synthetic arrays only: no model, real dataset or GPU access."""
import copy
import subprocess
import sys

import numpy as np
import pytest
import yaml
from PIL import Image

from src.common import digest_file, read_json, sample_output, write_json, write_npy
from src.datasets import cleargrasp as data
from src.datasets.cleargrasp_paths import (cleargrasp_prediction_dir, cleargrasp_figure_dir,
    full_sample_id, resolve_prediction_dir, load_result_manifest)
from src.evaluation.cleargrasp import saved_prediction
from src.evaluation.cleargrasp_analysis import analyze_saved_geometry, compare_raw_baseline
from src.evaluation.cleargrasp_boundary import boundary_geometry
from src.evaluation.metrics import MetricAccumulator
from src.evaluation.relative_geometry import relative_geometry, region_diagnostics, ordinal_result
from src.inference.da3_zero_shot import metric_depth_from_canonical
import src.cleargrasp_migration as migration


def saved_fixture(tmp_path, monkeypatch, count=2, legacy=True):
    root, output = tmp_path/'dataset', tmp_path/'output'
    folder = root/'real-test'/'d435'
    folder.mkdir(parents=True)
    shape = (40, 60)
    mask = np.zeros(shape, bool)
    mask[9:31, 18:42] = True
    gt = np.where(mask, 1., 2.).astype(np.float32)
    pred = 2*gt
    for n in range(count):
        number = f'{n:09}'
        Image.fromarray(np.full((*shape, 3), 90, np.uint8)).save(folder/f'{number}-transparent-rgb-img.jpg')
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/f'{number}-mask.png')
        (folder/f'{number}-opaque-depth-img.exr').write_bytes(b'synthetic EXR placeholder')
    (folder/'camera_intrinsics.yaml').write_text(yaml.safe_dump(dict(camera_name='D435',
        xres=60, yres=40, fx=600, fy=600, cx=30, cy=20)))
    monkeypatch.setattr(data, 'read_depth', lambda path: gt.copy())
    manifest = data.discover(root)
    write_json(output/'sample_manifest.json', manifest)
    for s in manifest['samples']:
        dest = sample_output(output, s['split'], s['sample_id']) if legacy else cleargrasp_prediction_dir(output, s)
        cal = data.calibration(root, s, shape)
        metric, conversion = metric_depth_from_canonical(pred/2, cal['K'], shape)
        write_npy(dest/'canonical_depth.npy', pred/2)
        write_npy(dest/'metric_depth.npy', metric)
        write_json(dest/'prediction.json', dict(sample_id=s['sample_id'], split=s['split'],
            metric_units='metres', metric_conversion=conversion, calibration=cal,
            input_shape=list(shape), input_sha256=digest_file(data.path_for(root, s, 'rgb')),
            output_sha256={f: digest_file(dest/f) for f in ('metric_depth.npy', 'canonical_depth.npy')}))
        fig = output/'figures'/s['split']/dest.name if legacy else cleargrasp_figure_dir(output, s)
        fig.mkdir(parents=True)
        (fig/'comparison.png').write_bytes(b'old comparison, must remain unchanged')
    baseline = dict(protocol='unscaled metric metres; finite positive GT AND prediction; no GT alignment',
        aggregation='global pixel weighted', regions={})
    for name, region in dict(full_image=np.ones(shape, bool), transparent=mask, background=~mask).items():
        acc = MetricAccumulator()
        for _ in range(count):
            acc.update(pred, gt, region)
        baseline['regions'][name] = acc.result()
    write_json(output/'metrics'/'raw_metric_summary.json', baseline)
    return root, output, manifest, gt, pred, mask


def snapshot(root):
    return {p.relative_to(root).as_posix(): digest_file(p) for p in root.rglob('*') if p.is_file()}


def test_readable_identity_unique_and_shared_paths_unchanged(tmp_path):
    samples = [dict(split=split, sample_id=f'{camera}/{number}')
        for split in ('real-test', 'real-val') for camera in ('d415', 'd435') for number in ('001', '002')]
    paths = [cleargrasp_prediction_dir(tmp_path, s) for s in samples]
    assert len(set(paths)) == 8
    assert paths[0] == tmp_path/'predictions'/'real-test'/'d415'/'001'
    assert len(sample_output(tmp_path, 'real-test', 'd415/001').name) == 24
    assert cleargrasp_prediction_dir(tmp_path, dict(split='real-test', sample_id='real-test/d415/001')) == paths[0]


@pytest.mark.parametrize('sid', ['../001', '/d415/001', 'C:/d415/001', r'd415\001',
    'd415/../001', 'd415/..', 'real-val/d415/001', 'd415/1:2', 'd415/001.', 'd415/１'])
def test_identity_rejects_unsafe_paths(tmp_path, sid):
    with pytest.raises(ValueError):
        cleargrasp_prediction_dir(tmp_path, dict(split='real-test', sample_id=sid))


def test_legacy_migration_dry_run_apply_idempotent(tmp_path, monkeypatch):
    root, output, manifest, _, pred, _ = saved_fixture(tmp_path, monkeypatch)
    before = snapshot(output)
    plan = migration.migrate_results(output, expected_samples=2)
    assert len(plan['operations']) == 4 and snapshot(output) == before
    assert plan['mapping'][0]['prediction_after'].endswith('d435/000000000')
    old = resolve_prediction_dir(output, manifest['samples'][0])
    np.testing.assert_array_equal(saved_prediction(output, manifest['samples'][0], True)[0], pred)
    result = migration.migrate_results(output, expected_samples=2, apply=True)
    assert result['state'] == 'completed' and not old.exists()
    for s in manifest['samples']:
        dest = cleargrasp_prediction_dir(output, s)
        assert resolve_prediction_dir(output, s) == dest
        np.testing.assert_array_equal(saved_prediction(output, s, True)[0], pred)
        assert (cleargrasp_figure_dir(output, s)/'comparison.png').read_bytes().startswith(b'old comparison')
    for op in plan['operations']:
        assert migration.tree_hashes(output, output/op['destination']) == op['sha256']
    assert digest_file(output/'sample_manifest.json') == before['sample_manifest.json']
    journal = digest_file(output/migration.JOURNAL)
    assert migration.migrate_results(output, expected_samples=2, apply=True)['state'] == 'already_readable'
    assert digest_file(output/migration.JOURNAL) == journal


def test_migration_checksum_conflict_and_count_abort_before_moves(tmp_path, monkeypatch):
    _, output, manifest, *_ = saved_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Expected all 20'):
        migration.migrate_results(output)
    s = manifest['samples'][-1]
    dest = resolve_prediction_dir(output, s)
    (dest/'metric_depth.npy').write_bytes(b'corrupt')
    before = snapshot(output)
    with pytest.raises(ValueError, match='integrity'):
        migration.migrate_results(output, expected_samples=2, apply=True)
    assert snapshot(output) == before
    assert not cleargrasp_prediction_dir(output, manifest['samples'][0]).exists()


def test_migration_refuses_existing_new_path(tmp_path, monkeypatch):
    _, output, manifest, *_ = saved_fixture(tmp_path, monkeypatch)
    cleargrasp_prediction_dir(output, manifest['samples'][0]).mkdir(parents=True)
    before = snapshot(output)
    with pytest.raises(ValueError, match='Ambiguous'):
        migration.migrate_results(output, expected_samples=2, apply=True)
    assert snapshot(output) == before


def test_migration_failure_rolls_back_and_interrupted_recovery(tmp_path, monkeypatch):
    _, output, _, *_ = saved_fixture(tmp_path, monkeypatch)
    before = snapshot(output)
    rename = migration._rename_checked
    calls = []
    def fail_once(*args):
        calls.append(args)
        if len(calls) == 2:
            raise OSError('injected failure')
        return rename(*args)
    monkeypatch.setattr(migration, '_rename_checked', fail_once)
    with pytest.raises(RuntimeError, match='rolled back'):
        migration.migrate_results(output, expected_samples=2, apply=True)
    assert read_json(output/migration.JOURNAL)['state'] == 'rolled_back'
    assert {k: v for k, v in snapshot(output).items() if k != migration.JOURNAL} == before
    monkeypatch.setattr(migration, '_rename_checked', rename)
    plan = migration.plan_migration(output, 2)
    plan['state'] = 'applying'
    write_json(output/migration.JOURNAL, plan)
    op = plan['operations'][0]
    rename(output, op['source'], op['destination'])  # crash before completed-index journal update
    assert migration.migrate_results(output, recover=True)['state'] == 'rolled_back'
    assert {k: v for k, v in snapshot(output).items() if k != migration.JOURNAL} == before


def test_duplicate_manifest_and_identity_mismatch(tmp_path, monkeypatch):
    _, output, manifest, *_ = saved_fixture(tmp_path, monkeypatch)
    duplicate = copy.deepcopy(manifest)
    duplicate['samples'][1] = copy.deepcopy(duplicate['samples'][0])
    duplicate['samples'][1]['sample_id'] = full_sample_id(duplicate['samples'][0])
    write_json(output/'sample_manifest.json', duplicate)
    with pytest.raises(ValueError, match='Duplicate'):
        load_result_manifest(output, 2)
    s = manifest['samples'][0]
    meta_path = resolve_prediction_dir(output, s)/'prediction.json'
    meta = read_json(meta_path)
    meta['sample_id'] = 'd415/000000000'
    write_json(meta_path, meta)
    with pytest.raises(ValueError, match='identity'):
        saved_prediction(output, s, True)


@pytest.mark.parametrize('field,value', [('metric_units', 'millimetres'), ('calibration', None),
    ('output_sha256', {}), ('metric_conversion', None)])
def test_strict_provenance_rejects_untrusted_arrays(tmp_path, monkeypatch, field, value):
    _, output, manifest, *_ = saved_fixture(tmp_path, monkeypatch, count=1)
    s = manifest['samples'][0]
    path = resolve_prediction_dir(output, s)/'prediction.json'
    meta = read_json(path)
    meta[field] = value
    write_json(path, meta)
    with pytest.raises(ValueError):
        saved_prediction(output, s, True)


def test_cpu_report_preserves_all_inputs_and_distinguishes_scale(tmp_path, monkeypatch):
    root, output, manifest, *_ = saved_fixture(tmp_path, monkeypatch)
    before, dataset_before = snapshot(output), snapshot(root)
    monkeypatch.setattr(data, 'discover', lambda *a, **kw: pytest.fail('Must not resample'))
    result = analyze_saved_geometry(output, root, expected_samples=2, pair_draws=3000,
        min_pairs=20, min_unique_pixels=5, ring_radius=15)
    assert result['status'] == 'complete' and result['analyzed_samples'] == 2
    assert result['baseline_comparison']['status'] == 'match'
    assert result['raw_metric']['regions']['full_image']['AbsRel'] == pytest.approx(1)
    assert result['diagnostic_metric']['oracle']['full_image']['AbsRel'] == pytest.approx(0)
    assert result['ordinal']['transparent_background']['accuracy'] == 1
    assert result['penetration']['error_rate'] == 0
    assert snapshot(root) == dataset_before
    assert {k: v for k, v in snapshot(output).items() if not k.startswith('geometry_analysis/')} == before
    report = output/'geometry_analysis'
    for file in ('summary.json', 'per_image.csv', 'scale_analysis.csv', 'ordinal_analysis.csv',
                 'penetration_analysis.csv', 'boundary_analysis.csv', 'report.md'):
        assert (report/file).is_file()
    for figure in ('scale_ratio_distribution', 'raw_vs_aligned_absrel', 'transparent_vs_background',
                   'ordinal_accuracy', 'penetration_rates', 'hardest_samples'):
        assert (report/'figures'/f'{figure}.png').is_file()
    assert full_sample_id(manifest['samples'][0]) in (report/'report.md').read_text(encoding='utf-8')
    assert 'GT-Dependent Diagnostic Only' in (report/'report.md').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        analyze_saved_geometry(output, root, expected_samples=2)


def test_all_twenty_validated_missing_one_no_partial_scores(tmp_path, monkeypatch):
    root, output, manifest, *_ = saved_fixture(tmp_path, monkeypatch, count=20)
    (resolve_prediction_dir(output, manifest['samples'][-1])/'metric_depth.npy').unlink()
    result = analyze_saved_geometry(output, root)
    assert result['status'] == 'validation_failed'
    assert (result['selected_samples'], result['verified_samples'], result['analyzed_samples']) == (20, 19, 0)
    assert 'raw_metric' not in result


def test_baseline_mismatch_is_reported_not_overwritten(tmp_path, monkeypatch):
    root, output, _, *_ = saved_fixture(tmp_path, monkeypatch, count=1, legacy=False)
    path = output/'metrics'/'raw_metric_summary.json'
    baseline = read_json(path)
    baseline['regions']['full_image']['AbsRel'] = .123
    write_json(path, baseline)
    before = digest_file(path)
    result = analyze_saved_geometry(output, root, expected_samples=1, pair_draws=100)
    assert result['status'] == 'baseline_mismatch'
    assert result['baseline_comparison']['differences'][0]['metric'] == 'AbsRel'
    assert digest_file(path) == before


def test_background_scale_ignores_transparent_gt_and_medians_are_separate():
    gt = np.array([[1., 1., 2., 2.]])
    mask = np.array([[True, True, False, False]])
    pred = np.array([[3., 3., 4., 4.]])
    result = region_diagnostics(pred, gt, mask)
    assert result['background_oracle_scale'] == .5
    assert result['regions']['transparent']['background_oracle_absrel'] == .5
    assert result['regions']['background']['background_oracle_absrel'] == 0
    changed = gt.copy()
    changed[mask] *= 10
    assert region_diagnostics(pred, changed, mask)['background_oracle_scale'] == .5
    result = region_diagnostics(np.array([[1., 3.]]), np.ones((1, 2)), np.ones((1, 2), bool))
    assert result['regions']['full_image']['median_pred_gt'] == 2
    assert result['oracle_scale'] == pytest.approx(2/3)


def test_order_and_boundary_scale_invariance_reversal_and_low_support():
    mask = np.zeros((40, 60), bool)
    mask[8:32, 16:44] = True
    gt = np.where(mask, 1., 2.)
    kwargs = dict(pair_draws=4000, ring_radius=15, min_pairs=20, min_unique_pixels=5)
    a = relative_geometry(gt, gt, mask, **kwargs)
    b = relative_geometry(gt*17, gt, mask, **kwargs)
    assert a['ordinal'] == b['ordinal'] and a['penetration'] == b['penetration']
    assert a['ordinal']['transparent_background']['accuracy'] == 1
    reversed_pred = np.where(mask, 3., 1.)
    wrong = relative_geometry(reversed_pred, gt, mask, **kwargs)
    assert wrong['penetration']['reversal_rate'] == 1
    correct_boundary = boundary_geometry(gt*17, gt, mask, 1/17, **kwargs)
    assert correct_boundary['contrast']['normalized_contrast_mae'] == pytest.approx(0)
    assert correct_boundary['contrast']['log_contrast_mae'] == pytest.approx(0)
    assert boundary_geometry(reversed_pred, gt, mask, 1., **kwargs)['contrast']['reversal_rate'] == 1
    for badmask in (np.zeros_like(mask), np.ones_like(mask), np.eye(40, 60, dtype=bool)):
        bad = boundary_geometry(gt, gt, badmask, 1., **kwargs)['contrast']
        assert bad['status'] == 'unavailable' and bad['reversal_rate'] is None
    pairs = np.tile([[0, 1]], (200, 1))
    result, _ = ordinal_result(np.array([[1., 2.]]), np.array([[1., 2.]]), pairs, .02, 10, 2)
    assert result['valid_pairs'] == 200 and result['unique_pairs'] == 1 and result['accuracy'] is None


def test_analysis_import_does_not_import_torch_or_da3():
    result = subprocess.run([sys.executable, '-c',
        "import sys; import src.evaluation.cleargrasp_analysis; "
        "assert 'torch' not in sys.modules; assert 'depth_anything_3' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
