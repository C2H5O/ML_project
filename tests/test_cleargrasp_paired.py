"""CPU-only synthetic paired experiment; official DA3 is never imported."""
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import yaml
from src.common import read_json, write_json, write_npy, digest_file, sample_output
from src.datasets import cleargrasp as data
from src.inference.da3_zero_shot import metric_depth_from_canonical, MODEL_ID, DA3_COMMIT
from src import cleargrasp_paired as paired
from src.cleargrasp_paired_config import DEFAULTS, parse_args
from src.evaluation.cleargrasp_paired import compare_pair, evaluate, differences, paired_statistics
from src.evaluation.paired_geometry import paired_relative_geometry, shared_ordinal_pairs


def scene():
    mask = np.zeros((30, 40), bool)
    mask[8:22, 10:30] = True
    gt = np.where(mask, 1., 2.) + np.arange(mask.size).reshape(mask.shape)*.00001
    return gt, mask


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    a = SimpleNamespace(**dict(DEFAULTS, data_root=tmp_path/'data', transparent_output=tmp_path/'transparent',
        output_dir=tmp_path/'paired', checkpoint=tmp_path/'checkpoint', expected_samples=2,
        pair_draws=500, bootstrap_draws=100, dry_run=False, resume=False, print_config=False, figure_dpi=35))
    a.checkpoint.mkdir()
    for name in ('config.json', 'model.safetensors'):
        (a.checkpoint/name).write_bytes(b'synthetic; never load')
    gt, mask = scene()
    for number in ('0001', '0002', '0003'):
        folder = a.data_root/'real-test/d415'
        folder.mkdir(parents=True, exist_ok=True)
        for material, color in (('transparent', 30), ('opaque', 180)):
            Image.fromarray(np.full((*gt.shape,3),color,np.uint8)).save(folder/f'{number}-{material}-rgb-img.jpg')
        Image.fromarray(mask.astype(np.uint8)*255).save(folder/f'{number}-mask.png')
        (folder/f'{number}-opaque-depth-img.exr').write_bytes(b'synthetic GT')
    (folder/'camera_intrinsics.yaml').write_text(yaml.safe_dump(dict(camera_name='D415', xres=40,yres=30,
        fx=300,fy=300,cx=20,cy=15)))
    monkeypatch.setattr(data, 'read_depth', lambda path: gt.astype(np.float32).copy())
    manifest = data.discover(a.data_root)
    manifest['samples'] = manifest['samples'][:2]  # Third sample must never be selected.
    for s in manifest['samples']:
        s['input_shape'] = list(gt.shape)
        s['input_sha256'] = digest_file(data.path_for(a.data_root,s,'rgb'))
        s['calibration'] = data.calibration(a.data_root,s,gt.shape)
        s['calibration_warning'] = None
    run = dict(model_id=MODEL_ID,da3_commit=DA3_COMMIT,checkpoint_hashes={n:digest_file(a.checkpoint/n)
        for n in ('config.json','model.safetensors')},process_res=560,process_res_method='upper_bound_resize',
        precision='bf16',device='cuda:0',offline=True,gt_input=False,selection=manifest['samples'])
    write_json(a.transparent_output/'sample_manifest.json',manifest)
    write_json(a.transparent_output/'run_metadata.json',dict(run_config=run))
    for i,s in enumerate(manifest['samples']):
        # One legacy and one new readable folder in the SAME source run.
        f = sample_output(a.transparent_output,s['split'],s['sample_id']) if i == 0 else paired.opaque_folder(a.transparent_output,s)
        metric, conversion = metric_depth_from_canonical(gt, s['calibration']['K'],gt.shape)
        write_npy(f/'metric_depth.npy',metric)
        write_json(f/'prediction.json',dict(sample_id=s['sample_id'],split=s['split'], input_shape=list(gt.shape),
            input_sha256=s['input_sha256'],calibration=s['calibration'], metric_conversion=conversion,
            metric_units='metres',run_config=run,output_sha256={'metric_depth.npy':digest_file(f/'metric_depth.npy')}))
    calls = []
    class FakePredictor:
        metadata = dict(synthetic=True)
        def __init__(self, **kwargs):
            calls.append('init')
        def predict_canonical(self,rgb,res,method):
            assert np.mean(rgb) == pytest.approx(180, abs=1)  # Opaque only.
            assert (res,method) == (560,'upper_bound_resize')
            calls.append('opaque')
            return gt.astype(np.float32),dict(seconds=.01,peak_cuda_allocated_bytes=0)
    monkeypatch.setattr(paired,'DA3Predictor',FakePredictor)
    return a,calls


def test_manifest_only_opaque_once_and_immutable_transparent(experiment):
    a,calls = experiment
    before = {str(p):digest_file(p) for p in a.transparent_output.rglob('*') if p.is_file()}
    assert paired.run(a) == 0
    assert calls == ['init','opaque','opaque']
    manifest = read_json(a.output_dir/'paired_manifest.json')
    assert len(manifest['records']) == 2
    assert [r['pair_id'] for r in manifest['records']] == ['real-test/d415/0001','real-test/d415/0002']
    for r in manifest['records']:
        t,o,gt,mask = paired.read_pair(a,r)
        np.testing.assert_equal(t,o)
        assert paired.opaque_folder(a.output_dir,r['sample']).name in ('0001','0002')
    with pytest.raises(FileExistsError):
        paired.run(a)
    a.resume = True
    assert paired.run(a) == 0
    assert calls == ['init','opaque','opaque']
    assert before == {str(p):digest_file(p) for p in a.transparent_output.rglob('*') if p.is_file()}


def test_dry_run_never_loads_or_writes(experiment):
    a,calls = experiment
    a.dry_run = True
    assert paired.run(a) == 0
    assert calls == [] and not a.output_dir.exists()


def test_missing_pair_no_replacement(experiment):
    a,calls = experiment
    (a.data_root/'real-test/d415/0001-opaque-rgb-img.jpg').unlink()
    assert paired.run(a) == 1
    m = read_json(a.output_dir/'paired_manifest.json')
    assert len(m['records']) == 2 and m['records'][0]['status'] == 'excluded'
    assert 'missing pair' in m['records'][0]['reason']
    assert calls == ['init','opaque']
    assert read_json(a.output_dir/'inference_summary.json')['excluded'] == 1


@pytest.mark.parametrize('bad', ['real-val/d415/0001','real-test/d435/0001','real-test/d415/0002'])
def test_cross_camera_split_number_rejected(experiment,bad):
    a,_ = experiment
    path = a.transparent_output/'sample_manifest.json'
    m = read_json(path)
    split,cam,num = bad.split('/')
    m['samples'][0].update(split=split,sample_id=f'{cam}/{num}',group=cam,number=num)
    write_json(path,m)
    with pytest.raises(ValueError):
        paired.prepare(a)


@pytest.mark.parametrize('kind', ['checkpoint','rgb','calibration','conversion','checksum','resolution','ambiguous'])
def test_preflight_rejects_protocol_or_source_changes(experiment,kind):
    a,calls = experiment
    s = read_json(a.transparent_output/'sample_manifest.json')['samples'][0]
    f = paired.source_folder(a.transparent_output,s)
    if kind == 'checkpoint':
        (a.checkpoint/'config.json').write_bytes(b'different')
        with pytest.raises(ValueError,match='checkpoint_hashes'):
            paired.prepare(a)
        return
    if kind == 'rgb':
        Image.fromarray(np.zeros((30,40,3),np.uint8)).save(data.path_for(a.data_root,s,'rgb'))
    elif kind == 'calibration':
        p = data.path_for(a.data_root,s,'intrinsics')
        d = yaml.safe_load(p.read_text()); d['fx'] = 301
        p.write_text(yaml.safe_dump(d))
    elif kind == 'checksum':
        write_npy(f/'metric_depth.npy',np.ones((30,40)))
    elif kind == 'conversion':
        d = read_json(f/'prediction.json'); d['metric_conversion']['gt_used'] = True
        write_json(f/'prediction.json',d)
    elif kind == 'resolution':
        Image.fromarray(np.zeros((10,20,3),np.uint8)).save(a.data_root/'real-test/d415/0001-opaque-rgb-img.jpg')
    else:
        new = paired.opaque_folder(a.transparent_output,s)
        write_json(new/'prediction.json',read_json(f/'prediction.json'))
    records = paired.prepare(a)['records']
    assert records[0]['status'] == 'excluded' and records[0]['reason']
    assert not calls


@pytest.mark.parametrize('which', ['source','inside_source','data','data_parent'])
def test_no_output_overlap(experiment,which):
    a,_ = experiment
    a.output_dir = dict(source=a.transparent_output,inside_source=a.transparent_output/'paired',
                       data=a.data_root,data_parent=a.data_root.parent)[which]
    with pytest.raises(ValueError):
        paired.prepare(a)


def test_raw_is_unaligned_and_common_intersection():
    gt,mask = scene()
    t,o = gt*3,gt*2
    t[0,0] = np.nan; o[0,1] = 0
    r = compare_pair(t,o,gt,mask,pair_draws=200)
    rows = [x for x in r['raw'] if x['sensitivity'] == 'original' and x['region'] == 'full_image']
    assert [x['AbsRel'] for x in rows] == pytest.approx([2.,1.])
    assert [x['valid_pixels'] for x in rows] == [gt.size-2]*2
    assert r['material'][0]['delta_material'] == pytest.approx(0,abs=1e-12)


def test_exact_shared_ordinal_endpoints_and_detects_geometry(monkeypatch):
    import src.evaluation.relative_geometry as module
    gt,mask = scene()
    t = np.where(mask,3.,gt); o = gt.copy()
    t[0,0] = np.nan; o[0,1] = np.nan
    seen = []
    original = module.ordinal_result
    def record(p,g,pairs,*args):
        seen.append(pairs.copy())
        return original(p,g,pairs,*args)
    monkeypatch.setattr(module,'ordinal_result',record)
    a,b = paired_relative_geometry(t,o,gt,mask,pair_draws=3000,seed=8)
    # First four calls accept pairs from GT, then exactly the same arrays score T/O.
    for left,right in zip(seen[4:8],seen[8:12]):
        np.testing.assert_array_equal(left,right)
    assert a['ordinal']['transparent_background']['accuracy'] == 0
    assert b['ordinal']['transparent_background']['accuracy'] == 1
    assert a['penetration']['reversal_rate'] == 1
    assert b['penetration']['reversal_rate'] == 0


def test_material_scale_invariance_and_equal_predictions():
    gt,mask = scene()
    t = np.where(mask,gt*3,gt*2)
    o = gt*4
    a = compare_pair(t,o,gt,mask,pair_draws=100)
    b = compare_pair(t*7,o*.3,gt,mask,pair_draws=100)
    for x,y in zip(a['material'],b['material']):
        assert x['delta_material'] == pytest.approx(y['delta_material'],abs=1e-12)
        assert x['delta_material'] == pytest.approx(np.log(1.5))
    same = compare_pair(t,t,gt,mask,pair_draws=100)
    assert all(x['delta_material'] == pytest.approx(0) for x in same['material'])
    assert same['raw'][0]['AbsRel'] == same['raw'][5]['AbsRel']


def test_registration_failure_skips_difference_and_verified_difference():
    gt,mask = scene()
    valid = np.ones(gt.shape,bool)
    assert set(differences(gt,gt,valid,~mask,False)) == {'status','reason'}
    d = differences(gt*2,gt,valid,~mask,True)
    np.testing.assert_allclose(d['background_referenced_log'],0,atol=1e-12)


def test_erosion_preserves_original_mask_and_reports_sensitivity():
    gt,mask = scene(); original = mask.copy()
    r = compare_pair(gt,gt,gt,mask,erosion_margin=2,pair_draws=100)
    np.testing.assert_equal(mask,original)
    assert r['material'][1]['object_pixels'] < r['material'][0]['object_pixels']


def test_bootstrap_and_worse_direction():
    s = paired_statistics([1,2,-1,0],42,500,['a','a','b','b'])
    assert s['transparent_worse'] == 2 and s['opaque_worse'] == 1 and s['ties'] == 1
    assert s == paired_statistics([1,2,-1,0],42,500,['a','a','b','b'])
    assert s['scene_bootstrap_95_ci'] is not None
    assert paired_statistics([1,2],42,20,['a',None])['scene_bootstrap_95_ci'] is None


def test_cpu_evaluation_and_visualization_and_resume(experiment):
    from src.visualization.cleargrasp_paired import visualize
    a,calls = experiment
    assert paired.run(a) == 0
    assert evaluate(a) == 0
    summary = read_json(a.output_dir/'metrics/paired_summary.json')
    assert summary['evaluated'] == 2 and summary['effective_independent_scenes'] is None
    assert summary['paired_metrics']['raw/original/full_image//AbsRel']['mean_difference'] == pytest.approx(0)
    assert visualize(a) == 0
    settings = read_json(a.output_dir/'figures/real-test/d415/0001/visualization.json')
    assert settings['per_pixel_difference_status'] == 'skipped'
    assert settings['depth_range_metres'] == [0,2]
    assert (a.output_dir/'contact_sheet.png').is_file()
    assert len(list((a.output_dir/'figures').rglob('*.png'))) == 12
    with pytest.raises(FileExistsError):
        evaluate(a)
    a.resume = True
    assert evaluate(a) == 0 and visualize(a) == 0
    assert calls == ['init','opaque','opaque']
    a.erosion_margin += 1
    with pytest.raises(ValueError,match='protocol differs'):
        evaluate(a)


def test_registration_approval_bound_to_images_and_scene(experiment):
    a,_ = experiment
    m = paired.prepare(a)
    rec = m['records'][0]
    review = dict(verified=True,reviewer='fixture',reason='synthetic exact geometry',scene='scene1',
        transparent_sha256=digest_file(data.path_for(a.data_root,rec['sample'],'rgb')),
        opaque_sha256=rec['opaque_sha256'],gt_sha256=rec['annotations']['gt_sha256'],
        mask_sha256=rec['annotations']['mask_sha256'])
    a.registration_review = a.output_dir.parent/'review.json'
    write_json(a.registration_review,{rec['pair_id']:review})
    assert paired.prepare(a)['records'][0]['registration']['verified'] is True
    review['opaque_sha256'] = '0'*64
    write_json(a.registration_review,{rec['pair_id']:review})
    assert paired.prepare(a)['records'][0]['status'] == 'excluded'


@pytest.mark.parametrize('update',[dict(process_res=448),dict(offline=False),dict(erosion_margin=-1),
    dict(seed=True),dict(depth_max=float('nan')),dict(unknown=1),dict(bootstrap_draws=0)])
def test_paired_config_validation(tmp_path,update):
    f = tmp_path/'config.yaml'
    f.write_text(yaml.safe_dump(dict(data_root=str(tmp_path),**update)))
    with pytest.raises(SystemExit):
        parse_args(['--config',str(f)])


def test_default_config_prints_without_dataset():
    a = parse_args(['--print-config'])
    assert a.data_root is None and a.expected_samples == 20


def test_atomic_resume_preserves_interrupted_staging(experiment):
    a,calls = experiment
    stranded = a.output_dir/'.staging/old-interrupted'
    stranded.mkdir(parents=True)
    (stranded/'opaque_metric_depth.npy').write_bytes(b'partial')
    # A real interrupted run has its immutable paired manifest already written.
    write_json(a.output_dir/'paired_manifest.json',paired.prepare(a))
    a.resume = True
    assert paired.run(a) == 0
    assert (stranded/'opaque_metric_depth.npy').read_bytes() == b'partial'
    assert calls == ['init','opaque','opaque']


def test_shared_pair_interface_rejects_bad_indices_and_honors_support():
    from src.evaluation.relative_geometry import relative_geometry
    gt,mask = scene()
    common = np.ones(gt.shape,bool)
    pairs = shared_ordinal_pairs(gt,mask,common,pair_draws=500,accepted=False)
    bad = {k:v.copy() for k,v in pairs.items()}
    bad['full_image'] = np.array([[0,gt.size]])
    with pytest.raises(ValueError,match='indices'):
        relative_geometry(gt,gt,mask,pair_indices=bad)
    a,b = paired_relative_geometry(gt,gt,gt,mask,pair_draws=500,min_pairs=100000)
    assert a['ordinal'] == b['ordinal']
    assert all(x['accuracy'] is None for x in a['ordinal'].values())


def test_paired_cpu_modules_do_not_import_torch():
    import subprocess
    import sys
    result = subprocess.run([sys.executable,'-c',
        'import src.evaluation.cleargrasp_paired; import src.visualization.cleargrasp_paired; '
        'import sys; assert "torch" not in sys.modules; assert "depth_anything_3" not in sys.modules'],
        capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_disjoint_support_is_excluded_instead_of_zero_error():
    gt,mask = scene()
    t,o = gt.copy(),gt.copy()
    t[:15] = np.nan; o[15:] = np.nan
    with pytest.raises(ValueError,match='No common'):
        compare_pair(t,o,gt,mask,pair_draws=100)
