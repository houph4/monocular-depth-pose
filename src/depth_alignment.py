"""Simple depth-adaptation controls. All observed and returned depths are metres."""
import numpy as np


def align_prediction(prediction, valid, sensor, anchor_mask, *,
                     representation='metric_z_meters', affine=True,
                     max_samples=5000, seed=0):
    prediction = np.asarray(prediction, np.float64)
    sensor = np.asarray(sensor, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(prediction)
    inverse = representation == 'relative_inverse_depth'
    if representation not in ('metric_z_meters', 'relative_inverse_depth'):
        raise ValueError(representation)
    if not inverse:
        valid &= prediction > 0
    fit = valid & np.asarray(anchor_mask, bool) & np.isfinite(sensor) & (sensor > 0)
    ids = np.flatnonzero(fit)
    empty = np.zeros(sensor.shape, np.float32)
    if len(ids) < 32:
        return empty, {'status': 'insufficient_anchors', 'anchors': int(len(ids))}
    if len(ids) > max_samples:
        ids = np.random.default_rng(seed).choice(ids, max_samples, replace=False)
    x, y = prediction.ravel()[ids], sensor.ravel()[ids]
    if inverse:
        y = 1. / y
    if affine:
        center, spread = np.median(x), np.std(x)
        if spread < 1e-8:
            return empty, {'status': 'constant_prediction', 'anchors': int(fit.sum())}
        design = np.column_stack(((x - center) / spread, np.ones_like(x)))
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        for _ in range(8):
            residual = design @ coef - y
            sigma = max(1e-6, 1.4826 * np.median(np.abs(residual - np.median(residual))))
            weight = np.minimum(1., 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
            root = np.sqrt(weight)
            coef = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
        scale = float(coef[0] / spread)
        shift = float(coef[1] - scale * center)
    else:
        eligible = np.abs(x) > 1e-8
        if eligible.sum() < 32:
            return empty, {'status': 'insufficient_nonzero_anchors', 'anchors': int(fit.sum())}
        scale = float(np.median(y[eligible] / x[eligible]))
        shift = 0.
    if not np.isfinite([scale, shift]).all() or scale <= 0:
        return empty, {'status': 'nonpositive_fit', 'scale': scale, 'shift': shift}
    corrected = scale * prediction + shift
    supported = valid & np.isfinite(corrected) & (corrected > 1e-8)
    if inverse:
        out = np.divide(1., corrected, out=np.zeros_like(corrected), where=supported)
    else:
        out = np.where(supported, corrected, 0.)
    out = out.astype(np.float32)
    out[~np.isfinite(out)] = 0.
    return out, {'status': 'ok', 'scale': scale, 'shift': shift,
                 'anchors': int(fit.sum()), 'fit_samples': int(len(ids)),
                 'fit_space': 'inverse_metres' if inverse else 'metres'}


def preserve_anchors(prediction, sensor):
    observed = np.isfinite(sensor) & (sensor > 0)
    return np.where(observed, sensor, prediction).astype(np.float32)
