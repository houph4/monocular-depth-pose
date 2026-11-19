# Adapting Monocular Depth Priors for Frozen RGB-D Object Pose Estimators

This repository contains the depth adapter used in the study by Pihong Hou, Yongfang Zhang, Boxuan Ye, and Jiadong Tan. It fits a monocular depth prior to sparse metric measurements and propagates the remaining depth residuals over the target mask. The adapter preserves measured depths and returns a dense camera-axis depth map for a frozen RGB-D pose estimator.

## Install

```sh
python -m pip install -r requirements.txt
```

## Adapt one observation

Create an input `.npz` with the following arrays on the same image grid.

| Array | Contents |
| --- | --- |
| `prior` | Monocular prediction, in camera-axis meters for metric priors |
| `sensor` | Available camera-axis measurements in meters, with zero at missing pixels |
| `mask` | Boolean target mask |
| `valid` | Optional Boolean validity mask for the prior |

```sh
python tools/adapt_depth.py input.npz output.npz --support target
```

The output contains `completed` and `aligned_prior` arrays in meters. The default target fit uses robust affine calibration and the scale-only fallback for degenerate fits. With fewer than eight valid anchors, the completion keeps the available measurements. The target protocol sets depth outside the target mask to zero.

Use `--support scene` to fit on all available scene measurements. This protocol requires at least 32 valid anchors for calibration and retains sensor depth outside the target. For a relative inverse-depth prior, use `--support scene --representation relative_inverse_depth`. In both protocols, local residual completion fills the target using eight nearest valid target anchors with inverse-distance weights.

The fitting implementations are in `src/depth_alignment.py` and `src/diagnostic_depth.py`. The `complete_target` function in `src/local_residual.py` also provides the spatial-interpolation baseline when called without a prior.

## Run the synthetic checks

```sh
python tools/check_depth_alignment.py
python tools/check_local_residual.py
```

The evaluation datasets are LM-O, TUD-L, and YCB-V from the [BOP benchmark](https://bop.felk.cvut.cz/datasets/).
