"""Paired target-local completion controls, with and without a monocular prior."""
import numpy as np
from scipy.spatial import cKDTree


def complete_target(sensor, mask, prior=None, neighbours=8):
    """Preserve measurements and fill missing target pixels by inverse-distance weights.

    With a prior, interpolate sensor-minus-prior residuals. Without a prior,
    interpolate measured depth. Identical spatial neighbours and weights isolate
    the contribution of the frozen monocular geometry. This is a feasibility
    control, not a claimed new depth-completion algorithm.
    """
    sensor = np.asarray(sensor, np.float32)
    mask = np.asarray(mask, bool)
    anchors = mask & np.isfinite(sensor) & (sensor > 0)
    if prior is not None:
        prior = np.asarray(prior, np.float32)
        prior_valid = np.isfinite(prior) & (prior > 0)
        anchors &= prior_valid
    else:
        prior_valid = np.ones_like(mask)
    query = mask & ~(np.isfinite(sensor) & (sensor > 0)) & prior_valid
    result = sensor.copy()
    n = int(anchors.sum())
    if n < 8 or not query.any():
        return result, dict(anchors=n, filled=0, status='insufficient_support' if n < 8 else 'no_holes')
    reference_yx = np.column_stack(np.nonzero(anchors))
    query_yx = np.column_stack(np.nonzero(query))
    distance, index = cKDTree(reference_yx).query(query_yx, k=min(neighbours, n), workers=1)
    weight = 1. / np.maximum(distance, .5)**2
    weight /= weight.sum(1, keepdims=True)
    values = sensor[anchors].astype(np.float64)
    if prior is not None:
        values -= prior[anchors]
    filled = (weight * values[index]).sum(1)
    if prior is not None:
        filled += prior[query]
    result[query] = np.where(np.isfinite(filled) & (filled > 0), filled, 0)
    return result, dict(anchors=n, filled=int(np.count_nonzero(result[query])),
                        median_nearest_distance_px=float(np.median(distance[:, 0])),
                        p95_nearest_distance_px=float(np.quantile(distance[:, 0], .95)),
                        neighbours=neighbours, status='ok')
