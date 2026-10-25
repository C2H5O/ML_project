import numpy as np
import pytest
from src.evaluation.metrics import MetricAccumulator, align_scale, region_masks


def test_known_metrics_and_strict_delta_threshold():
    gt = np.array([[1.,2.,4.]])
    pred = np.array([[1.,2.5,2.]])
    result = MetricAccumulator().update(pred, gt).result()
    assert result["AbsRel"] == pytest.approx(.25)
    assert result["RMSE"] == pytest.approx(np.sqrt(4.25/3))
    assert result["delta1"] == pytest.approx(1/3)


def test_invalid_and_empty():
    gt = np.array([[1.,0.,np.nan,np.inf,-1.]])
    pred = np.ones_like(gt)
    assert MetricAccumulator().update(pred,gt).result()["valid_pixels"] == 1
    result = MetricAccumulator().update(pred,gt,np.zeros_like(gt,bool)).result()
    assert result["status"] == "unavailable" and result["AbsRel"] is None
    with pytest.raises(ValueError, match="No valid"):
        align_scale(pred,gt,np.zeros_like(gt,bool))


def test_alignment_recovers_scale_and_shared_region_protocol():
    gt = np.array([[1.,2.,3.,4.]])
    raw = gt / 5
    aligned, scale = align_scale(raw,gt)
    assert scale == pytest.approx(5)
    assert MetricAccumulator().update(aligned,gt).result()["RMSE"] == pytest.approx(0)
    assert np.array_equal(raw,gt/5)
    masks = region_masks(np.array([[0,1,2,3]]),gt.shape)
    assert np.array_equal(masks["target_classes_2_3"], [[False,False,True,True]])
    assert np.array_equal(masks["ordinary_class_0"], [[True,False,False,False]])


def test_abnormal_scale():
    with pytest.raises(ValueError, match="Abnormal"):
        align_scale(np.full((2,2),1e-12),np.ones((2,2)))
    with pytest.raises(ValueError):
        align_scale(np.zeros((2,2)), np.ones((2,2)))


def test_global_pixel_aggregation_not_image_mean_rmse():
    a = MetricAccumulator()
    a.update(np.array([[2.]]),np.array([[1.]]))
    a.update(np.ones((1,3)),np.ones((1,3)))
    result = a.result()
    assert result["AbsRel"] == .25
    assert result["RMSE"] == .5
    assert result["delta1"] == .75
    assert result["valid_pixels"] == 4 and result["samples"] == 2


def test_invalid_predictions_excluded_and_missing_regions_unavailable():
    p = np.array([[1.,0.,np.nan,np.inf,-1.]])
    assert MetricAccumulator().update(p,np.ones_like(p)).result()["valid_pixels"] == 1
    masks = region_masks(None,p.shape)
    assert not masks["target_classes_2_3"].any()
    assert masks["full_image"].all()
