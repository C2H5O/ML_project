"""Mask-distance regions and scale-invariant local boundary contrasts."""
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.evaluation.metrics import MetricAccumulator, valid_pixels
from src.evaluation.relative_geometry import local_pairs, ordinal_result, log_residual_statistics


def boundary_regions(mask, width=4, radius=30, margin=2):
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not 0 <= margin < width <= radius:
        raise ValueError('Boundary parameters require 0 <= margin < width <= radius')
    inside = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    outside = distance_transform_edt(~mask)
    # Image borders are not observed object/background boundaries.
    yy, xx = np.indices(mask.shape)
    observed = (yy >= width) & (xx >= width) & (yy < mask.shape[0]-width) & (xx < mask.shape[1]-width)
    return dict(interior=mask & (inside > width),
                boundary=mask & (inside > margin) & (inside <= width) & observed,
                nearby_background=~mask & (outside > margin) & (outside <= radius))


def boundary_geometry(pred, gt, mask, background_scale, seed=42, pair_draws=20000,
                      gt_gap_m=.02, ring_radius=30, boundary_margin=2, boundary_width=4,
                      min_pairs=50, min_unique_pixels=10, min_region_pixels=10, return_details=False):
    if min(pair_draws, min_pairs, min_unique_pixels, min_region_pixels) < 1 or gt_gap_m <= 0:
        raise ValueError('Invalid boundary support/sampling parameters')
    p, g, mask = np.asarray(pred, float), np.asarray(gt, float), np.asarray(mask, bool)
    valid = valid_pixels(p, g, mask=np.ones(mask.shape, bool))
    regions = boundary_regions(mask, boundary_width, ring_radius, boundary_margin)
    stats = {}
    for name, region in regions.items():
        v = valid & region
        stats[name] = dict(mask_pixels=int(region.sum()), valid_pixels=int(v.sum()),
            raw=MetricAccumulator().update(p, g, region).result(),
            background_aligned_absrel=MetricAccumulator().update(p*background_scale, g, region).result()['AbsRel']
                if background_scale is not None else None,
            **log_residual_statistics(p, g, v))
    reliable = mask.any() and (~mask).any() and all(s['valid_pixels'] >= min_region_pixels for s in stats.values())
    pairs = np.empty((0, 2), np.int64)
    if reliable:
        pairs = local_pairs(mask, valid, pair_draws, np.random.default_rng(seed), ring_radius,
                            boundary_margin, source_region=regions['boundary'])
    candidates = len(pairs)
    support, pairs = ordinal_result(p, g, pairs, gt_gap_m, min_pairs, min_unique_pixels)
    contrast = dict(**support, normalized_contrast_mae=None, log_contrast_mae=None,
                    reversal_rate=None, tie_rate=None, candidate_draws=pair_draws,
                    spatial_candidates=candidates, accepted_draw_fraction=len(pairs)/pair_draws)
    if not reliable:
        contrast['reason'] = 'Insufficient valid interior/boundary/background ring, or missing object/background mask'
    normalized_error, log_error = np.empty(0), np.empty(0)
    if len(pairs):
        a, b = pairs.T
        pa, pb, ga, gb = p.ravel()[a], p.ravel()[b], g.ravel()[a], g.ravel()[b]
        pc, gc = 2*(pb-pa)/(pb+pa), 2*(gb-ga)/(gb+ga)
        normalized_error = np.abs(pc-gc)
        log_error = np.abs((np.log(pb)-np.log(pa)) - (np.log(gb)-np.log(ga)))
        if contrast['status'] == 'available':
            contrast.update(normalized_contrast_mae=float(normalized_error.mean()),
                log_contrast_mae=float(log_error.mean()), reversal_rate=float(np.mean(pc*gc < 0)),
                tie_rate=float(np.mean(pc == 0)))
    result = dict(regions=stats, contrast=contrast, protocol=dict(boundary_width_px=boundary_width,
        exclusion_margin_px=boundary_margin, ring_radius_px=ring_radius, gt_gap_m=gt_gap_m,
        min_region_pixels=min_region_pixels, min_unique_pairs=min_pairs, min_unique_pixels=min_unique_pixels,
        normalized_contrast='2*(B-T)/(B+T)', log_contrast='ln(B)-ln(T)',
        interpretation='local image-plane contrast; not a measured occluded background on the same camera ray'))
    if return_details:
        result['_details'] = dict(pairs=pairs, normalized_error=normalized_error, log_error=log_error,
                                  region_masks=regions)
    return result
