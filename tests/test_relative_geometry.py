import numpy as np
import pytest
from src.evaluation.relative_geometry import (relative_geometry, oracle_scale,
                                              region_diagnostics, ordinal_result)


def scene():
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    gt = np.where(mask, 1., 2.)
    return gt, mask


def test_scale_bias_is_not_penetration():
    gt, mask = scene()
    r = relative_geometry(gt*7, gt, mask, pair_draws=10000, ring_radius=15)
    assert r['ordinal']['transparent_background']['valid_pairs'] > 100
    assert r['ordinal']['transparent_background']['accuracy'] == 1
    assert r['penetration']['error_rate'] == 0
    d = r['diagnostics']['regions']['full_image']
    assert d['oracle_absrel'] == pytest.approx(0)
    assert d['silog'] == pytest.approx(0, abs=1e-12)
    assert d['median_log_scale_bias'] == pytest.approx(np.log(7))


@pytest.mark.parametrize('surface,expected_reversal', [(3., True), (2., False), (1.95, False)])
def test_penetration_reverse_and_compression(surface, expected_reversal):
    gt, mask = scene()
    p = np.where(mask, surface, 2.)
    r = relative_geometry(p, gt, mask, pair_draws=5000, ring_radius=15)
    assert r['penetration']['error_rate'] == 1
    assert r['penetration']['reversal_rate'] == float(expected_reversal)
    assert r['penetration']['compression_rate'] == float(not expected_reversal)


def test_background_in_front_is_not_a_penetration_reference():
    gt, mask = scene()
    gt = np.where(mask, 2., 1.)
    r = relative_geometry(gt, gt, mask, pair_draws=5000, ring_radius=15)
    assert r['ordinal']['transparent_background']['accuracy'] == 1
    assert r['penetration']['status'] == 'unavailable'


def test_insufficient_background_and_gt():
    gt = np.ones((10, 10))
    mask = np.ones(gt.shape, bool)
    r = relative_geometry(gt, gt, mask)
    assert r['penetration']['valid_pairs'] == 0
    assert r['diagnostics']['background_oracle_scale'] is None
    r = relative_geometry(gt, np.zeros_like(gt), mask)
    assert all(o['status'] == 'unavailable' for o in r['ordinal'].values())


def test_oracle_is_median_ratios_not_ratio_medians():
    p = np.array([[1., 2., 3.]])
    g = np.array([[3., 100., 1.]])
    assert oracle_scale(p, g, np.ones(p.shape, bool)) == 3
    assert np.median(g)/np.median(p) == 1.5


def test_background_oracle_does_not_hide_surface_bias():
    gt, mask = scene()
    p = np.where(mask, gt*4, gt*2)
    r = region_diagnostics(p, gt, mask)
    assert r['background_oracle_scale'] == .5
    assert r['regions']['transparent']['background_oracle_absrel'] == 1


def test_seed_and_scale_invariance():
    gt, mask = scene()
    gt = gt + np.arange(gt.size).reshape(gt.shape)/10000
    p = np.where(mask, 1.8, 2.)
    a = relative_geometry(p, gt, mask, pair_draws=3000)
    b = relative_geometry(p, gt, mask, pair_draws=3000)
    c = relative_geometry(p*13, gt, mask, pair_draws=3000)
    assert a == b
    assert a['ordinal'] == c['ordinal']
    assert a['penetration'] == c['penetration']


def test_threshold_strict_and_prediction_ties_are_incorrect():
    p = np.array([[1., 1., 4.]])
    g = np.array([[1., 2., 3.]])
    pairs = np.array([[0, 1], [0, 2]])
    r, _ = ordinal_result(p, g, pairs, 1.)
    assert r['valid_pairs'] == 1 and r['accuracy'] == 1
    r, _ = ordinal_result(p, g, pairs, .5)
    assert r['accuracy'] == .5


def test_invalid_predictions_excluded():
    gt, mask = scene()
    p = gt.copy()
    p[mask] = np.nan
    r = relative_geometry(p, gt, mask)
    assert r['ordinal']['transparent_background']['status'] == 'unavailable'
    assert r['diagnostics']['regions']['transparent']['valid_pixels'] == 0
