"""Check unit/representation contracts before the downstream pilot."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from depth_alignment import align_prediction, preserve_anchors

rng = np.random.default_rng(42)
truth = rng.uniform(.4, 3., (30, 40)).astype(np.float32)
anchors = rng.random(truth.shape) < .25
sensor = np.where(anchors, truth, 0.)
for representation, prediction in [
    ('metric_z_meters', (truth - .1) / 1.4),
    ('relative_inverse_depth', (1. / truth - .08) / 2.3),
]:
    result, info = align_prediction(prediction, np.ones_like(anchors), sensor, anchors,
                                    representation=representation)
    np.testing.assert_allclose(result, truth, atol=2e-6)
    assert info['status'] == 'ok'
    fused = preserve_anchors(result + .2, sensor)
    np.testing.assert_array_equal(fused[anchors], sensor[anchors].astype(np.float32))
    rejected, info = align_prediction(prediction, anchors, sensor, np.zeros_like(anchors),
                                      representation=representation)
    assert not rejected.any() and info['status'] == 'insufficient_anchors'
print('PASS: metric/inverse-depth affine recovery, anchor preservation, insufficient support')
