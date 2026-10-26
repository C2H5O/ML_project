"""Paired entrypoint for the existing deterministic relative-geometry sampler."""
import numpy as np
from src.evaluation.metrics import valid_pixels
from src.evaluation.relative_geometry import relative_geometry


def shared_ordinal_pairs(gt, mask, common, seed=42, pair_draws=20000, gt_gap_m=.02,
                         ring_radius=30, boundary_margin=2, accepted=True, **protocol):
    """Draw and accept once, using GT and common support only."""
    result = relative_geometry(np.where(common, gt, np.nan), gt, mask, seed=seed,
        pair_draws=pair_draws, gt_gap_m=gt_gap_m, ring_radius=ring_radius,
        boundary_margin=boundary_margin, return_details=True, **protocol)
    return result['_details']['ordinal_pairs' if accepted else 'candidate_pairs']


def paired_relative_geometry(transparent, opaque, gt, mask, **protocol):
    """Draw once, then pass identical candidates to both geometry evaluations.

    Common validity and GT-only acceptance guarantee identical accepted pairs
    too. No oracle scale enters ordinal or penetration calculations.
    """
    common = valid_pixels(transparent, gt) & valid_pixels(opaque, gt)
    protocol.setdefault('seed', 42)
    pairs = shared_ordinal_pairs(gt, mask, common, accepted=False, **protocol)
    return tuple(relative_geometry(np.where(common, p, np.nan), gt, mask,
        pair_indices=pairs, **protocol) for p in (transparent, opaque))
