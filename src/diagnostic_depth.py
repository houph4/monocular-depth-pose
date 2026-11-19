"""Controlled metric-depth interventions; no pose labels enter these functions."""
import hashlib
import numpy as np
from local_residual import complete_target


def keyed_seed(key, repeat=0):
    return int.from_bytes(hashlib.sha256(f'{key}:anchors:{repeat}'.encode()).digest()[:4], 'little')


def metric_fit(prior, valid, sensor, anchors, affine=True, max_samples=5000, seed=0):
    prior = np.asarray(prior, np.float64)
    valid = np.asarray(valid, bool) & np.isfinite(prior) & (prior > 0)
    fit = valid & anchors & np.isfinite(sensor) & (sensor > 0)
    ids = np.flatnonzero(fit)
    info = dict(anchors=int(len(ids)), requested_model='affine' if affine else 'scale')
    if len(ids) < 8:
        return np.zeros_like(sensor, dtype=np.float32), dict(**info, status='insufficient_anchors')
    if len(ids) > max_samples:
        ids = np.random.default_rng(seed).choice(ids, max_samples, replace=False)
    x, y = prior.ravel()[ids], np.asarray(sensor).ravel()[ids]
    scale, shift = float(np.median(y/x)), 0.
    status = 'scale_fit'
    if affine and np.std(x) > 1e-8:
        center, spread = np.median(x), np.std(x)
        A = np.column_stack(((x-center)/spread, np.ones_like(x)))
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        for _ in range(8):
            error = A @ coef-y
            sigma = max(1e-6, 1.4826*np.median(np.abs(error-np.median(error))))
            w = np.sqrt(np.minimum(1., 1.345*sigma/np.maximum(np.abs(error), 1e-12)))
            coef = np.linalg.lstsq(A*w[:, None], y*w, rcond=None)[0]
        a, b = float(coef[0]/spread), float(coef[1]-coef[0]*center/spread)
        if np.isfinite([a,b]).all() and a > 0:
            scale, shift, status = a, b, 'affine_fit'
        else:
            status = 'scale_fallback_nonpositive_affine'
    elif affine:
        status = 'scale_fallback_constant_prior'
    corrected = scale*prior+shift
    good = valid & np.isfinite(corrected) & (corrected > 0)
    return np.where(good, corrected, 0).astype(np.float32), dict(
        **info, status=status, scale=scale, shift_m=shift, fit_samples=len(ids))


def sample_target(sensor, mask, key, budget, repeat):
    ids = np.flatnonzero(mask & np.isfinite(sensor) & (sensor > 0))
    order = np.random.default_rng(keyed_seed(key, repeat)).permutation(ids)
    chosen = order[:budget]
    out = np.zeros_like(sensor, dtype=np.float32)
    out.ravel()[chosen] = sensor.ravel()[chosen]
    return out, dict(requested=budget, actual=len(chosen), repeat=repeat,
                     selection_seed=keyed_seed(key,repeat),
                     anchor_ids_sha256=hashlib.sha256(chosen.tobytes()).hexdigest())


def shape_control(depth, mask, mode, key):
    out = depth.copy()
    valid = mask & np.isfinite(depth) & (depth > 0)
    if valid.any():
        if mode == 'median':
            out[valid] = np.median(depth[valid])
        elif mode == 'shuffle':
            out[valid] = np.random.default_rng(keyed_seed(key, 999)).permutation(depth[valid])
        else:
            raise ValueError(mode)
    return out


def budget_inputs(prior, valid, sensor, mask, key, budget, repeat):
    sparse, info = sample_target(sensor, mask, key, budget, repeat)
    corrected, fit = metric_fit(prior, valid, sparse, sparse > 0, seed=keyed_seed(key,repeat))
    corrected = np.where(mask, corrected, 0).astype(np.float32)
    fused = np.where(sparse > 0, sparse, corrected).astype(np.float32)
    spatial, spatial_info = complete_target(sparse, mask)
    residual, residual_info = complete_target(sparse, mask, corrected)
    inputs = dict(sensor_sparse=sparse, spatial=spatial, calibration_only=corrected,
                  calibration_preserve=fused, residual=residual)
    assert all(np.count_nonzero(x[~mask]) == 0 for x in inputs.values())
    for name in ['spatial','calibration_preserve','residual']:
        assert np.array_equal(inputs[name][sparse > 0], sparse[sparse > 0])
    return inputs, dict(**info, fit=fit, spatial=spatial_info,
                       residual=residual_info), sparse > 0


def depth_metrics(depth, reference, roi, exclude=None):
    ref = roi & np.isfinite(reference) & (reference > 0)
    if exclude is not None:
        ref &= ~exclude
    good = ref & np.isfinite(depth) & (depth > 0)
    n = int(good.sum())
    result = dict(reference_pixels=int(ref.sum()), valid_pixels=n,
                  coverage=float(n/max(1,ref.sum())))
    if not n:
        return dict(**result, mae_mm=None, rmse_mm=None, signed_bias_mm=None,
                    absrel=None, median_ratio=None)
    err = depth[good].astype(np.float64)-reference[good]
    return dict(**result, mae_mm=float(np.mean(np.abs(err))*1000),
                rmse_mm=float(np.sqrt(np.mean(err**2))*1000),
                signed_bias_mm=float(np.mean(err)*1000),
                absrel=float(np.mean(np.abs(err)/reference[good])),
                median_ratio=float(np.median(depth[good]/reference[good])))
