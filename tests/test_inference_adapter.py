import types
import numpy as np
import pytest
from src.inference.da3_zero_shot import DA3Predictor, metric_flag, precision_for_device


def test_official_missing_metric_flag_is_unknown():
    assert metric_flag({}) is None
    assert metric_flag(0) == 0
    assert metric_flag(1) == 1
    assert metric_flag(2) is None


def test_adapter_official_call_is_rgb_only_and_handles_missing_metric_flag():
    import torch
    seen = {}
    class API:
        def inference(self, **kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(depth=np.ones((1,2,3),np.float32),is_metric={},
                                         processed_images=np.zeros((1,2,3,3),np.uint8))
    predictor=DA3Predictor.__new__(DA3Predictor)
    predictor.device=torch.device('cpu')
    predictor.model=API()
    rgb=np.zeros((2,3,3),np.uint8)
    canonical,native,info=predictor.predict(rgb,np.diag([300.,300.,1.]),448)
    assert native.shape == (2,3) and info['prediction_is_metric'] is None
    assert set(seen) == {'image','process_res','process_res_method','infer_gs','export_dir'}
    assert len(seen['image']) == 1 and seen['image'][0] is rgb
    assert not seen['infer_gs'] and seen['export_dir'] is None
    with pytest.raises(ValueError,match='Crop'):
        predictor.predict(rgb,np.eye(3),448,'upper_bound_crop')


def test_cpu_precision_explicit():
    import torch
    device=torch.device('cpu')
    assert precision_for_device(torch,device,'auto') == 'fp32'
    with pytest.raises(ValueError,match='CPU'):
        precision_for_device(torch,device,'fp16')
