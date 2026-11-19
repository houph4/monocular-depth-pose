"""Convert one registered monocular prior and sparse measurement map to metric depth."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from depth_alignment import align_prediction
from diagnostic_depth import metric_fit
from local_residual import complete_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='NPZ file with prior, sensor, mask, and optional valid arrays')
    parser.add_argument('output', type=Path, help='Output NPZ file')
    parser.add_argument('--support', choices=('target', 'scene'), default='target')
    parser.add_argument('--representation', choices=('metric_z_meters', 'relative_inverse_depth'),
                        default='metric_z_meters')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.support == 'target' and args.representation != 'metric_z_meters':
        parser.error('The fixed-budget target fit uses metric priors. Use --support scene for a relative prior.')
    if args.output.suffix.lower() != '.npz':
        parser.error('The output path must end in .npz.')
    if args.input.resolve() == args.output.resolve():
        parser.error('Choose a separate output path.')

    with np.load(args.input, allow_pickle=False) as inputs:
        prior = inputs['prior']
        sensor = inputs['sensor']
        mask = inputs['mask'].astype(bool)
        valid = inputs['valid'].astype(bool) if 'valid' in inputs else np.isfinite(prior)
    if prior.ndim != 2 or any(array.shape != prior.shape for array in (sensor, mask, valid)):
        parser.error('prior, sensor, mask, and valid must use the same two-dimensional image grid.')

    observed = np.isfinite(sensor) & (sensor > 0)
    if args.support == 'target':
        observed &= mask
    sensor = np.where(observed, sensor, 0).astype(np.float32)
    if args.support == 'target':
        aligned, fit = metric_fit(prior, valid, sensor, observed, seed=args.seed)
        aligned = np.where(mask, aligned, 0).astype(np.float32)
    else:
        aligned, fit = align_prediction(prior, valid, sensor, observed,
                                        representation=args.representation, seed=args.seed)
    completed, completion = complete_target(sensor, mask, aligned)
    np.savez_compressed(args.output, completed=completed, aligned_prior=aligned)
    print(f"Calibration {fit['status']}. Filled {completion['filled']} target pixels.")


if __name__ == '__main__':
    main()
