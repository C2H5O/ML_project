import numpy as np
import pytest
from src.inference.da3_zero_shot import metric_depth_from_canonical
from src.runtime import runtime_issues, require_gpu_name


def test_focal_scaling_uses_native_resolution_and_both_focals():
    K = np.array([[1000.,0.,280.],[0.,800.,224.],[0.,0.,1.]])
    canonical = np.full((364,448),2.,np.float32)
    metric, info = metric_depth_from_canonical(canonical,K,(448,560))
    # Actual anisotropic resize after patch rounding: sx=.8, sy=.8125.
    assert info['focal_native_px'] == pytest.approx((800+650)/2)
    assert info['multiplier'] == pytest.approx(725/300)
    np.testing.assert_allclose(metric, 2*725/300)
    assert info['gt_used'] is False and info['applied_exactly_once']
    np.testing.assert_array_equal(canonical, np.full((364,448),2.))


@pytest.mark.parametrize('K',[None,np.zeros((3,3)),np.diag([-1,300,1]),np.full((3,3),np.nan)])
def test_unconfirmed_intrinsics_block_scaling(K):
    with pytest.raises(ValueError,match='intrinsics'):
        metric_depth_from_canonical(np.ones((2,2)),K,(2,2))


def test_existing_runtime_bounds_allow_reuse_without_exact_version_forcing():
    assert runtime_issues((3,10,20),'2.3.1+cu121') == []
    assert runtime_issues((3,10,12),'2.2.2+cu118') == []
    assert runtime_issues((3,11,0),'2.3.1')
    assert runtime_issues((3,10,20),'2.4.0')


def test_required_gpu_rejects_4090_without_d_and_laptop():
    import types
    class Cuda:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def get_device_name(device): return 'NVIDIA GeForce RTX 4090'
    torch=types.SimpleNamespace(cuda=Cuda)
    with pytest.raises(RuntimeError,match='Required GPU'):
        require_gpu_name(torch,types.SimpleNamespace(type='cuda'),'RTX 4090D')


@pytest.mark.parametrize('process_res,expected_shape',[(448,(364,448)),(560,(448,560))])
def test_official_processor_and_official_scaling_match_on_realistic_K(process_res,expected_shape):
    # Source-only integration test; no weights or network. Skip if the existing
    # runtime has no official source, rather than installing anything for tests.
    pytest.importorskip('depth_anything_3.utils.io.input_processor')
    import torch
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from depth_anything_3.utils.alignment import apply_metric_scaling
    rgb=np.zeros((448,560,3),np.uint8)
    K=np.array([[705.8,0,280.1],[0,706.0,223.5],[0,0,1]],np.float64)
    images, _, native_K=InputProcessor()([rgb],intrinsics=K[None],process_res=process_res,
                                       process_res_method='upper_bound_resize',sequential=True)
    shape=images.shape[-2:]
    assert tuple(shape) == expected_shape
    canonical=np.full(shape,1.2,np.float32)
    metric,info=metric_depth_from_canonical(canonical,K,rgb.shape[:2])
    np.testing.assert_allclose(info['native_K'],native_K[0].numpy(),rtol=1e-6)
    official=apply_metric_scaling(torch.from_numpy(canonical)[None,None],native_K[None])[0,0].numpy()
    np.testing.assert_allclose(metric,official,rtol=1e-6)
