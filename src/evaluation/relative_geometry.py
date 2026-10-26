"""RGB prediction diagnostics. GT alignment never modifies saved predictions."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from src.evaluation.metrics import MetricAccumulator, valid_pixels

ORACLE_LABEL = 'GT-dependent oracle diagnostic, not deployable metric performance.'
REGIONS = ('full_image', 'transparent', 'background')


def oracle_scale(pred, gt, valid):
    """Median of pixelwise ratios (NOT ratio of medians)."""
    if not valid.any():
        return None
    return float(np.median(np.asarray(gt, float)[valid] / np.asarray(pred, float)[valid]))


def region_diagnostics(pred, gt, mask):
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    valid = valid_pixels(p, g)
    if np.shape(mask) != g.shape:
        raise ValueError('Mask/GT shape mismatch')
    regions = dict(full_image=np.ones(g.shape, bool), transparent=mask, background=~mask)
    scale = oracle_scale(p, g, valid)
    bg_scale = oracle_scale(p, g, valid & ~mask)
    results = {}
    for name, region in regions.items():
        v = valid & region
        n = int(v.sum())
        if not n:
            results[name] = dict(status='unavailable', valid_pixels=0, oracle_absrel=None,
                                 background_oracle_absrel=None, median_log_scale_bias=None,
                                 silog=None, log_residual_mad=None)
            continue
        e = np.log(p[v]) - np.log(g[v])
        bias = float(np.median(e))
        results[name] = dict(status='available', valid_pixels=n,
            oracle_absrel=float(np.mean(np.abs(scale * p[v] - g[v]) / g[v])),
            background_oracle_absrel=(float(np.mean(np.abs(bg_scale * p[v] - g[v]) / g[v]))
                                      if bg_scale is not None else None),
            median_log_scale_bias=bias, silog=float(np.sqrt(np.mean((e - e.mean()) ** 2))),
            log_residual_mad=float(np.median(np.abs(e - bias))))
    return dict(label=ORACLE_LABEL, oracle_scale=scale, background_oracle_scale=bg_scale,
                silog_definition='sqrt(mean((log(pred/GT)-mean(log(pred/GT)))**2)); natural log, no x100',
                dispersion_definition='median absolute deviation of log residual from its median', regions=results)


def _uniform_pairs(rng, first, second, count):
    if len(first) == 0 or len(second) == 0:
        return np.empty((0, 2), np.int64)
    return np.column_stack((rng.choice(first, count), rng.choice(second, count)))


def local_pairs(mask, valid, count, rng, radius=30, margin=2):
    """Bounded Monte Carlo local pairs, with replacement; no all-pairs matrix.

    Transparent source is eroded; target must be outside the mask, within the
    radius and beyond a boundary margin. No distant-background fallback.
    """
    h, w = mask.shape
    interior = binary_erosion(mask, iterations=margin) if margin else mask
    distance = distance_transform_edt(~mask)
    ring = ~mask & valid & (distance > margin) & (distance <= radius)
    first = np.flatnonzero(interior & valid)
    if not len(first) or not ring.any():
        return np.empty((0, 2), np.int64)
    # Exactly count candidate draws, not count accepted pairs; coverage is reported.
    a = rng.choice(first, count)
    dy, dx = rng.integers(-radius, radius + 1, size=(2, count))
    y, x = a // w + dy, a % w + dx
    inside = (y >= 0) & (y < h) & (x >= 0) & (x < w) & (dx*dx + dy*dy <= radius*radius)
    a, y, x = a[inside], y[inside], x[inside]
    good = ring[y, x]
    return np.column_stack((a[good], (y*w+x)[good]))


def ordinal_result(pred, gt, pairs, threshold):
    p, g = pred.ravel(), gt.ravel()
    if len(pairs):
        pairs = pairs[(pairs[:, 0] != pairs[:, 1]) &
                      (np.abs(g[pairs[:, 0]] - g[pairs[:, 1]]) > threshold)]
    n = len(pairs)
    correct = int(np.sum(np.sign(p[pairs[:, 0]] - p[pairs[:, 1]]) ==
                         np.sign(g[pairs[:, 0]] - g[pairs[:, 1]]))) if n else 0
    return dict(status='available' if n else 'unavailable', valid_pairs=n, correct_pairs=correct,
                accuracy=correct / n if n else None), pairs


def relative_geometry(pred, gt, mask, seed=42, pair_draws=20000, gt_gap_m=0.02,
                      ring_radius=30, boundary_margin=2, compression_ratio=0.25):
    if (pair_draws < 1 or gt_gap_m <= 0 or ring_radius < 1 or boundary_margin < 0 or
            not 0 < compression_ratio < 1 or boundary_margin >= ring_radius):
        raise ValueError('Invalid relative geometry protocol parameters')
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    v = valid_pixels(p, g)
    mask = np.asarray(mask, bool)
    if mask.shape != g.shape:
        raise ValueError('Mask shape mismatch')
    rng = np.random.default_rng(seed)
    t, b, f = [np.flatnonzero(x & v) for x in (mask, ~mask, np.ones(mask.shape, bool))]
    tb = local_pairs(mask, v, pair_draws, rng, ring_radius, boundary_margin)
    pairs = dict(full_image=_uniform_pairs(rng, f, f, pair_draws),
                 transparent_transparent=_uniform_pairs(rng, t, t, pair_draws),
                 transparent_background=tb,
                 background_background=_uniform_pairs(rng, b, b, pair_draws))
    ordinal, accepted = {}, {}
    for name, pair in pairs.items():
        ordinal[name], accepted[name] = ordinal_result(p, g, pair, gt_gap_m)
    pairs = accepted['transparent_background']
    if len(pairs):
        pairs = pairs[g.ravel()[pairs[:, 1]] - g.ravel()[pairs[:, 0]] > gt_gap_m]
    n = len(pairs)
    penetration = dict(status='unavailable', valid_pairs=n, reversed_pairs=0, compressed_pairs=0,
                       error_pairs=0, error_rate=None, reversal_rate=None, compression_rate=None,
                       reason='No reliable local pairs with GT transparent surface in front of background')
    if n:
        a, b = pairs.T
        gt_log_gap = np.log(g.ravel()[b]) - np.log(g.ravel()[a])
        pred_log_gap = np.log(p.ravel()[b]) - np.log(p.ravel()[a])
        reversed_order = pred_log_gap < 0
        compressed = (pred_log_gap >= 0) & (pred_log_gap < compression_ratio * gt_log_gap)
        penetration.update(status='available', reason=None,
            reversed_pairs=int(reversed_order.sum()), compressed_pairs=int(compressed.sum()),
            error_pairs=int((reversed_order | compressed).sum()),
            error_rate=float(np.mean(reversed_order | compressed)),
            reversal_rate=float(reversed_order.mean()), compression_rate=float(compressed.mean()))
    return dict(diagnostics=region_diagnostics(p, g, mask), ordinal=ordinal, penetration=penetration,
                protocol=dict(seed=int(seed), pair_draws=pair_draws, sampling='with replacement; bounded candidate draws',
                              gt_gap_m=gt_gap_m, ring_radius_px=ring_radius,
                              boundary_margin_px=boundary_margin, compression_ratio=compression_ratio,
                              penetration_definition='GT front pairs: reversed or pred log(bg/T) < ratio * GT log(bg/T)',
                              gt_scaling_for_ordinal_or_penetration=False))
