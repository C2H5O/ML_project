# DA3METRIC-LARGE protocol

The only active model is depth-anything/DA3METRIC-LARGE. Only Raw Metric
Evaluation is accepted; no scale/shift fitting to GT is performed. Active server
commands are in [README](../README.md), using existing environment vggtomast3r.
No environment is created, no torch/dependency is installed/upgraded, and
setup_env.sh now performs read-only checks only.

## Verified source path

Pinned official source: 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4.

- configs/da3metric-large.yaml: standalone ViT-L + DPT, no camera decoder/encoder.
- model/da3.py: standalone DepthAnything3Net does not invoke apply_metric_scaling.
- utils/alignment.py: apply_metric_scaling multiplies by mean(fx,fy)/300.
- api.py: inference does not perform focal conversion when extrinsics are absent;
  its pose alignment branch must not be used to manufacture metric scale.
- InputProcessor: scales K for boundary resize and patch-divisibility resize.
  The final user-confirmed process_res=560 preserves the saved 448x560 grid.
  The superseded process_res=448 case maps to 364x448 and is regression-tested.

See the [official FAQ](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/README.md)
and [scaling code](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/utils/alignment.py).

The implementation preserves canonical network output, transforms saved calibrated
K using actual native/output dimensions, converts once to metres, then resizes
only predictions back to the saved GT grid. It does not pass GT or masks to the
model. Calibration K is metadata for output conversion, not a GT-derived scale.
Unexpected is_metric=1 from the standalone pinned implementation is rejected.
GT is used only for validity/errors, never to alter a prediction.

Model weights are float32, with an instance-local AMP wrapper to honour BF16/FP16
or explicit FP32. The underlying official network call/preprocessing/output logic
is preserved. The retained lazy-export patch tool is not executed automatically;
existing working server code is not silently patched. A source-commit mismatch
fails before a reported experiment can be accepted.

## Data and material definitions

The saved Booster depth is already metric. Original disparity was converted by
the existing preprocessing using mtxL and its recorded stereo baseline, then
resized/cropped with a corresponding K update. Do not repeat those operations.
Use saved K, not K_original, as the input-grid calibration for the DA3 resize.

Existing mtxL/projection focal discrepancies remain warnings. The dataset audit
verifies K transform consistency, not independent physical calibration accuracy.
Predictions and evaluation use the supplied calibration convention unchanged.

The [Booster paper](https://arxiv.org/html/2301.08245v3) defines severity classes:
0 opaque/diffuse; 1 strongly distorted reflection/transmission; 2 partly distorted
recognizable appearance; 3 regular reflection/transmission. Target=2+3 and
ordinary=0 are declared project groups; other=0+1 and each class are also saved.
These labels do not separate transparent versus mirror semantics.

Only processed train is currently available. Keep balanced/unbalanced observations
of the same scene together in future splits. Raw directory names overlap at
Cooker1 across train/test; resolve that warning before any future fine-tuning
partition. No future training or hyperparameter search is implemented here.
Head-only and LoRA + Head on A800 80GB must share the later total <=60h budget.

## Reproducibility and failure boundaries

Record model Hub SHA or local checkpoint hashes, official code commit, environment
versions, device/GPU, precision, actual native dimensions, saved/native K,
calibration hashes, multiplier and no-GT-alignment flags. The run configuration
cannot be mixed with prior BASE outputs or a different precision/resolution.
Resume validates RGB, calibration, config and all numerical array hashes.
Evaluation recomputes the calibration-only conversion to verify provenance.

The tests include analytical focal cases, anisotropic patch rounding, missing K,
comparison against official InputProcessor and apply_metric_scaling, no-double-
scaling verification, GT perturbations that change errors without changing depth,
material metrics, empty regions, raw output formats, resume and GPU-name guard.
Tests use synthetic arrays. All 76 real Booster calibrations were independently
checked against official preprocessing/scaling without a downloaded checkpoint.
There is no remote server execution because SSH is prohibited.
