from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from local_residual import complete_target

y, x = np.indices((40, 40))
truth = (1 + .001*x + .002*y).astype(np.float32)
keep = (x % 5 == 0) & (y % 5 == 0)
sensor = np.where(keep, truth, 0)
mask = np.ones_like(keep)
prior = truth + .13
output, info = complete_target(sensor, mask, prior)
assert np.array_equal(output[keep], sensor[keep])
assert np.max(np.abs(output-truth)) < 3e-7
plain, _ = complete_target(sensor, mask)
constant, _ = complete_target(sensor, mask, np.ones_like(sensor))
assert np.allclose(plain, constant, atol=2e-7, rtol=0)
empty, diag = complete_target(np.zeros_like(sensor), mask, prior)
assert not empty.any() and diag['status'] == 'insufficient_support'
print('PASS: exact measurements; offset recovery; constant-prior control; unsupported targets')
