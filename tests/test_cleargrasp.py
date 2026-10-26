"""CPU-only fixtures; never import official DA3 or load a real checkpoint."""
import importlib.util
from pathlib import Path
import sys
import types
import os
import numpy as np
from PIL import Image
import pytest
import yaml
from src.common import read_json, write_json, write_npy, digest_file, sample_output
from src.datasets import cleargrasp as data
from src.evaluation.cleargrasp import evaluate_saved
from src.inference.da3_zero_shot import metric_depth_from_canonical
from src.visualization.cleargrasp import visualize_saved


def fixture(root, numbers=('0001', '0002')):
    folder = root / 'real-test' / 'd435'
    folder.mkdir(parents=True)
    shape = (20, 30)
    for number in numbers:
        Image.fromarray(np.full((*shape, 3), 90, np.uint8)).save(folder / f'{number}-transparent-rgb-img.jpg')
        mask = np.zeros(shape, np.uint8)
        mask[5:15, 10:20] = 255
        Image.fromarray(mask).save(folder / f'{number}-mask.png')
        (folder / f'{number}-opaque-depth-img.exr').write_bytes(b'fixture')
    (folder / 'camera_intrinsics.yaml').write_text(yaml.safe_dump(dict(camera_name='D435',
        xres=30, yres=20, fx=600, fy=600, cx=15, cy=10)))
    return folder


def load_runner():
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location('cleargrasp_runner', Path(scripts)/'run_cleargrasp_zero_shot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_pairing_not_sorted_positions(tmp_path):
    folder = fixture(tmp_path)
    (folder / '0001-mask.png').rename(folder / '0003-mask.png')
    manifest = data.discover(tmp_path)
    s = manifest['samples'][0]
    assert s['number'] == '0001' and s['missing'] == ['mask']
    assert len(manifest['orphan_annotations']) == 1
    assert manifest['splits']['real-val']['present'] is False
    with pytest.raises(ValueError, match='unavailable'):
        data.select_samples(manifest, 'real-val')


def test_synthetic_pairing(tmp_path):
    group = tmp_path/'synthetic-val'/'glass'
    for folder, name in [('rgb-imgs', '42-rgb.jpg'), ('depth-imgs-rectified', '42-depth-rectified.exr'),
                         ('segmentation-masks', '42-segmentation-mask.png')]:
        p = group/folder/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'fixture')
    m = data.discover(tmp_path)
    assert len(m['samples']) == 1 and m['samples'][0]['missing'] == []
    assert data.calibration(tmp_path, m['samples'][0], (20, 30)) is None


def test_calibration_checks_and_metric_grid(tmp_path):
    fixture(tmp_path)
    s = data.discover(tmp_path)['samples'][0]
    c = data.calibration(tmp_path, s, (20, 30))
    p, info = metric_depth_from_canonical(np.ones((10, 15)), c['K'], (20, 30))
    assert np.all(p == 1)
    assert info['native_K'][0][0] == 300 and info['multiplier'] == 1
    with pytest.raises(ValueError, match='dimensions'):
        data.calibration(tmp_path, s, (10, 15))
    s['group'] = 'd415'
    with pytest.raises(ValueError, match='camera'):
        data.calibration(tmp_path, s, (20, 30))


def test_annotations_do_not_use_sensor_depth_or_resize(tmp_path, monkeypatch):
    folder = fixture(tmp_path)
    s = data.discover(tmp_path)['samples'][0]
    called = []
    def reader(path):
        called.append(path.name)
        return np.ones((20, 30), np.float32)
    monkeypatch.setattr(data, 'read_depth', reader)
    gt, mask, info = data.annotations(tmp_path, s, (20, 30))
    assert called == ['0001-opaque-depth-img.exr']
    assert mask.sum() == 100 and info['gt_units'] == 'metres'
    with pytest.raises(ValueError, match='match RGB'):
        data.annotations(tmp_path, s, (10, 15))


def test_sampling_and_source_protection(tmp_path):
    fixture(tmp_path)
    m = data.discover(tmp_path)
    assert data.select_samples(m, 'real-test', 1) == data.select_samples(m, 'real-test', 1)
    with pytest.raises(ValueError, match='read-only'):
        data.inspect_dataset(tmp_path, tmp_path/'results')


def test_opencv_reads_red_not_blue(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'OpenEXR', None)
    arr = np.zeros((2, 3, 3), np.float32)
    arr[..., 0], arr[..., 2] = 99, 2
    monkeypatch.setitem(sys.modules, 'cv2', types.SimpleNamespace(IMREAD_UNCHANGED=-1, imread=lambda *a: arr))
    assert np.all(data.read_depth(tmp_path/'test.exr') == 2)


def test_missing_exr_backend_diagnostic(monkeypatch, tmp_path):
    for name in ('OpenEXR', 'cv2'):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(RuntimeError, match='no automatic installation'):
        data.read_depth(tmp_path/'test.exr')


def test_offline_pipeline_resume_evaluation_and_figures(tmp_path, monkeypatch):
    root, output = tmp_path/'dataset', tmp_path/'output'
    fixture(root)
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    for n in ('config.json', 'model.safetensors'):
        (checkpoint/n).write_bytes(b'test fixture only')
    module = load_runner()
    calls = []
    environment_before = dict(os.environ)
    class Predictor:
        metadata = dict(test_fixture=True)
        def __init__(self, **kwargs):
            calls.append('init')
        def predict_canonical(self, rgb, resolution, method):
            assert rgb.shape == (20, 30, 3) and rgb.dtype == np.uint8
            calls.append('predict')
            return np.ones((10, 15), np.float32), dict(seconds=.01, peak_cuda_allocated_bytes=0)
    monkeypatch.setattr(module, 'DA3Predictor', Predictor)
    monkeypatch.setattr(data, 'read_depth', lambda p: np.ones((20, 30), np.float32))
    config = tmp_path/'run.yaml'
    config.write_text(yaml.safe_dump(dict(data_root=str(root), checkpoint=str(checkpoint),
        output_dir=str(output), all=True, skip_visualization=True, pair_draws=100)))
    args = module.parser().parse_args(['--config', str(config)])
    assert module.run(args) == 0
    assert dict(os.environ) == environment_before
    assert calls == ['init', 'predict', 'predict']
    assert module.run(args) == 0
    assert calls == ['init', 'predict', 'predict']
    assert read_json(output/'metrics/raw_metric_summary.json')['regions']['full_image']['AbsRel'] == 0
    assert visualize_saved(output) == []
    assert (output/'contact_sheet.png').is_file()
    assert len(list((output/'figures').rglob('*.png'))) == 16
    args.process_res = 448
    with pytest.raises(ValueError, match='differs'):
        module.run(args)


def test_canonical_only_never_reports_raw_metric(tmp_path, monkeypatch):
    root, out = tmp_path/'data', tmp_path/'out'
    fixture(root, ('0001',))
    m = data.discover(root)
    write_json(out/'sample_manifest.json', m)
    s = m['samples'][0]
    folder = sample_output(out, s['split'], s['sample_id'])
    write_npy(folder/'canonical_depth.npy', np.ones((10, 15)))
    write_json(folder/'prediction.json', dict(sample_id=s['sample_id'], split=s['split'], metric_conversion=None,
        input_sha256=digest_file(data.path_for(root, s, 'rgb')),
        output_sha256={'canonical_depth.npy': digest_file(folder/'canonical_depth.npy')}))
    monkeypatch.setattr(data, 'read_depth', lambda p: np.ones((20, 30)))
    assert evaluate_saved(out) == []
    raw = read_json(out/'metrics/raw_metric_summary.json')
    assert raw['regions']['full_image']['AbsRel'] is None
    relative = read_json(out/'metrics/relative_geometry_summary.json')
    assert relative['regions']['full_image']['oracle_absrel'] == 0


def test_shared_config_paths_independent_of_cwd(tmp_path, monkeypatch):
    from src.cleargrasp_config import PROJECT_ROOT
    module = load_runner()
    config = tmp_path/'paths.yaml'
    config.write_text(yaml.safe_dump(dict(data_root='data/cleargrasp', checkpoint='checkpoints/DA3METRIC-LARGE',
        output_dir='results/cleargrasp_test', process_res=448, all=True)))
    monkeypatch.chdir(tmp_path)
    args = module.parser().parse_args(['--config', str(config), '--process-res', '560', '--max-samples', '3'])
    assert args.data_root == PROJECT_ROOT/'data/cleargrasp'
    assert args.checkpoint == PROJECT_ROOT/'checkpoints/DA3METRIC-LARGE'
    assert args.output_dir == PROJECT_ROOT/'results/cleargrasp_test'
    assert args.process_res == 560 and args.max_samples == 3 and not args.all


@pytest.mark.parametrize('updates', [dict(unknown_key=1), dict(offline=False), dict(all='false'),
    dict(max_samples=0), dict(seed=True), dict(gt_gap_m=float('nan')), dict(precision='bad'),
    dict(data_root='$CLEARGRASP_ROOT'), dict(data_root='%CLEARGRASP_ROOT%'),
    dict(compression_ratio=1.), dict(boundary_margin=30)])
def test_invalid_config_rejected(tmp_path, updates):
    from src.cleargrasp_config import load_config
    config = tmp_path/'bad.yaml'
    config.write_text(yaml.safe_dump(dict(data_root=str(tmp_path), checkpoint='checkpoints/local',
        output_dir='results/test', **{k:v for k,v in updates.items() if k not in ('data_root',)})))
    if 'data_root' in updates:
        config.write_text(yaml.safe_dump(updates))
    with pytest.raises(ValueError):
        load_config(config)


@pytest.mark.parametrize('option', ['--data-root', '--checkpoint', '--output-dir'])
def test_paths_cannot_be_overridden_on_cli(tmp_path, option):
    module = load_runner()
    config = tmp_path/'paths.yaml'
    config.write_text(yaml.safe_dump(dict(data_root=str(tmp_path), checkpoint='checkpoints/local',
                                         output_dir='results/test')))
    with pytest.raises(SystemExit):
        module.parser().parse_args(['--config', str(config), option, '/unapproved-path'])


def test_unconfigured_server_root_prints_but_cannot_run():
    module = load_runner()
    assert module.parser().parse_args(['--print-config']).data_root is None
    with pytest.raises(SystemExit):
        module.parser().parse_args([])


@pytest.mark.parametrize('script_name,function_name', [
    ('inspect_cleargrasp', 'inspect_dataset'), ('evaluate_cleargrasp', 'evaluate_saved'),
    ('visualize_cleargrasp', 'visualize_saved')])
def test_all_entrypoints_use_yaml_paths(tmp_path, monkeypatch, script_name, function_name):
    load_runner()  # Add scripts/ to import path.
    import importlib
    module = importlib.import_module(script_name)
    config = tmp_path/'shared.yaml'
    root, out = tmp_path/'data', tmp_path/'results'
    config.write_text(yaml.safe_dump(dict(data_root=str(root), output_dir=str(out),
        checkpoint='checkpoints/local', offline=True, split='real-test', pair_draws=123)))
    seen = []
    def fake(*args, **kwargs):
        seen.append((args, kwargs))
        if script_name == 'inspect_cleargrasp':
            return dict(data_root=str(root), splits={}, exr_capabilities={}, missing_files=[],
                        unreadable_files=[], orphan_annotations=[])
        return []
    monkeypatch.setattr(module, function_name, fake)
    monkeypatch.setattr(sys, 'argv', [script_name, '--config', str(config)])
    assert module.main() == 0
    if script_name == 'inspect_cleargrasp':
        assert seen[0][0] == (root, out)
    else:
        assert seen[0][0] == (out,)
        assert seen[0][1]['data_root'] == root


def test_exr_reader_does_not_change_environment(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'OpenEXR', None)
    monkeypatch.setitem(sys.modules, 'cv2', types.SimpleNamespace(IMREAD_UNCHANGED=-1,
        imread=lambda *args: np.ones((2, 3), np.float32)))
    before = dict(os.environ)
    data.read_depth(tmp_path/'depth.exr')
    assert dict(os.environ) == before


def test_modern_openexr_reads_r_without_imath(tmp_path, monkeypatch):
    expected = np.array([[1., 2., 0.], [np.nan, 4., np.inf]], np.float16)
    seen = []
    class File:
        parts = [object()]
        def __init__(self, path, separate_channels):
            seen.append((path, separate_channels))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def channels(self):
            return {'R': types.SimpleNamespace(pixels=expected),
                    'B': types.SimpleNamespace(pixels=np.full(expected.shape, 99))}
    monkeypatch.setitem(sys.modules, 'OpenEXR', types.SimpleNamespace(File=File))
    monkeypatch.setitem(sys.modules, 'Imath', None)
    before = dict(os.environ)
    result = data.read_depth(tmp_path/'test.exr')
    np.testing.assert_equal(result, expected.astype(np.float32))
    assert result.dtype == np.float32 and seen[0][1] is True
    assert dict(os.environ) == before


def test_legacy_openexr_reader_still_supported(tmp_path, monkeypatch):
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    closed = []
    class InputFile:
        def __init__(self, path):
            pass
        def header(self):
            return {'dataWindow': types.SimpleNamespace(min=types.SimpleNamespace(x=0,y=0),
                                                        max=types.SimpleNamespace(x=2,y=1))}
        def channel(self, name, pixel_type):
            assert name == 'R'
            return expected.tobytes()
        def close(self):
            closed.append(True)
    class PixelType:
        FLOAT = 2
        def __init__(self, value):
            pass
    monkeypatch.setitem(sys.modules, 'OpenEXR', types.SimpleNamespace(InputFile=InputFile))
    monkeypatch.setitem(sys.modules, 'Imath', types.SimpleNamespace(PixelType=PixelType))
    np.testing.assert_array_equal(data.read_depth(tmp_path/'test.exr'), expected)
    assert closed == [True]


def test_disabled_opencv_is_not_retried_for_every_file(tmp_path, monkeypatch):
    calls = []
    def disabled(*args):
        calls.append(args)
        raise RuntimeError('imgcodecs: OpenEXR codec is disabled')
    backend = types.SimpleNamespace(IMREAD_UNCHANGED=-1, imread=disabled)
    monkeypatch.setitem(sys.modules, 'OpenEXR', None)
    monkeypatch.setitem(sys.modules, 'cv2', backend)
    monkeypatch.setattr(data, '_opencv_exr_disabled', None)
    for name in ('one.exr', 'two.exr'):
        with pytest.raises(RuntimeError, match='server vggtomast3r'):
            data.read_depth(tmp_path/name)
    assert len(calls) == 1


def smoke_fixture(tmp_path):
    root, output = tmp_path/'dataset', tmp_path/'output'
    fixture(root, tuple(f'{i:04d}' for i in range(23)))
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    for name in ('config.json', 'model.safetensors'):
        (checkpoint/name).write_bytes(b'fixture only')
    config = tmp_path/'smoke.yaml'
    config.write_text(yaml.safe_dump(dict(data_root=str(root), output_dir=str(output),
        checkpoint=str(checkpoint), seed=42, max_samples=1, all=True,
        skip_evaluation=True, skip_visualization=True)))
    return root, output, config


def test_smoke_pipeline_forces_20_and_all_stages(tmp_path, monkeypatch):
    root, output, config = smoke_fixture(tmp_path)
    module = load_runner()
    calls = []
    def annotation(root, sample, shape):
        calls.append('preflight')
        return np.ones(shape), np.zeros(shape, bool), {}
    class Predictor:
        metadata = dict(test_fixture=True)
        def __init__(self, **kwargs):
            assert calls == ['preflight']*20
            calls.append('init')
        def predict_canonical(self, *args):
            calls.append('predict')
            return np.ones((10, 15), np.float32), dict(seconds=.01, peak_cuda_allocated_bytes=0)
    monkeypatch.setattr(module, 'annotations', annotation)
    monkeypatch.setattr(module, 'DA3Predictor', Predictor)
    monkeypatch.setattr(module, 'evaluate_saved', lambda *a, **k: calls.append('evaluate') or [])
    monkeypatch.setattr(module, 'visualize_saved', lambda *a, **k: calls.append('visualize') or [])
    args = module.parser().parse_args(['--config', str(config), '--smoke-test'])
    before = dict(os.environ)
    assert module.run(args) == 0
    assert calls == ['preflight']*20 + ['init'] + ['predict']*20 + ['evaluate', 'visualize']
    assert dict(os.environ) == before
    manifest = read_json(output/'sample_manifest.json')
    expected = data.select_samples(data.discover(root), 'real-test', 20, 42)
    assert [s['sample_id'] for s in manifest['samples']] == [s['sample_id'] for s in expected]
    assert len({s['sample_id'] for s in manifest['samples']}) == 20
    assert read_json(output/'smoke_preflight.json')['selected'] == 20


def test_smoke_gt_failure_stops_before_model_loading(tmp_path, monkeypatch):
    _, output, config = smoke_fixture(tmp_path)
    module = load_runner()
    def broken(*args):
        raise RuntimeError('EXR reader unavailable')
    monkeypatch.setattr(module, 'annotations', broken)
    monkeypatch.setattr(module, 'DA3Predictor', lambda **kw: pytest.fail('GPU model must not be loaded'))
    args = module.parser().parse_args(['--config', str(config), '--smoke-test'])
    with pytest.raises(RuntimeError, match='no GPU model was loaded'):
        module.run(args)
    report = read_json(output/'smoke_preflight.json')
    assert len(report['records']) == 20
    assert all(r['status'] == 'failed' for r in report['records'])
    assert not (output/'predictions').exists()


def test_smoke_requires_exactly_20_available_samples(tmp_path):
    root, _, config = smoke_fixture(tmp_path)
    # Restrict discovery to a smaller fixture without changing the dataset files.
    module = load_runner()
    from unittest.mock import patch
    manifest = data.discover(root)
    manifest['samples'] = manifest['samples'][:2]
    with patch.object(module, 'discover', return_value=manifest):
        args = module.parser().parse_args(['--config', str(config), '--smoke-test'])
        with pytest.raises(ValueError, match='requires 20 unique RGB samples'):
            module.run(args)
