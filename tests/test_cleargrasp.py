"""CPU-only fixtures; never import official DA3 or load a real checkpoint."""
import importlib.util
from pathlib import Path
import sys
import types
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
    args = module.parser().parse_args(['--data-root', str(root), '--checkpoint', str(checkpoint),
        '--output-dir', str(output), '--all', '--skip-visualization', '--pair-draws', '100'])
    assert module.run(args) == 0
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
