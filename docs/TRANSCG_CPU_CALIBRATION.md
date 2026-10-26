# Calibrate existing TransCG predictions on CPU

This is a postprocessing workflow, not a new DA3 experiment. It reuses exactly
the images in the saved `sample_manifest.json`; it never samples or runs the
model. No torch, official DA3 package, HF client or GPU import is required.
Existing `results/transcg_zero_shot/` and Booster results remain read-only.

## Why the original figures are canonical-only

The original configuration had `intrinsics: null` and `intrinsics_shape: null`.
The network outputs therefore have no verified metric scale. Readable GT in
metres does not establish a prediction scale. The additional camera-info package
must supply the correct D435 K and a confirmed pixel grid; its download alone
does not prove either. `calibration_root` is independent from the existing RGB
`data_root`, which must not be changed just to find camera information.

## Inspect on the actual Linux server

The server camera-info directory and original 20 predictions are not available
in the development workspace. Do not fill guessed K or 720x1280 in the config.
Use the existing environment, without installs or weight loading:

```bash
conda activate vggtomast3r
cd /public/home/2024141520249/Documents/Projects/ML_pro/ML_project

python scripts/inspect_transcg_calibration.py \
    --config configs/transcg_zero_shot.yaml
```

The configured `calibration_root` is
`/public/home/2024141520249/Documents/Projects/ML_pro/data/transcg`.
The inspector searches recursively for the exact camera-specific official
`1-camIntrinsics-D435.npy` basename. It reports all matches, raw matrix
shape/dtype/numeric values, validation errors, SHA256, and **actual dimensions
of every selected RGB**, verified against the original manifest/hash. It also
reports GT encoding/units and observed mask values/counts. It does not write
files. stdout is JSON; explanatory messages go to stderr. Its exit status is 1
until reliable K and a user-confirmed grid agree with all selected RGBs.

It distinguishes K from poses by camera filename, numeric 3x3 pinhole structure,
finite positive focal lengths and homogeneous row, and rejects 4x4/3x4 poses,
rotation matrices and explicit pose/extrinsics filenames. Principal point must
lie inside the declared grid. Automatic discovery never searches arbitrary
`.npy` files as candidate K. Multiple D435 matches require an explicit path.
An explicit custom filename still requires the user's camera/grid confirmation;
matrix structure alone cannot prove physical calibration provenance.

After checking the official calibration provenance, fill these two fields in
`configs/transcg_zero_shot.yaml`:

```yaml
intrinsics: /actual/verified/path/1-camIntrinsics-D435.npy
intrinsics_shape: [ACTUAL_HEIGHT, ACTUAL_WIDTH]
```

Replace the placeholders with real integer dimensions. The inspector's observed
RGB size is evidence about the image, not proof that a bare K was calibrated on
that grid. Verify both. This implementation requires the declared K grid to
equal the original saved RGB grid; mismatches fail instead of inventing an
additional crop/resize transform. The first inspection prints all 20 dimensions
without requiring you to enter them first. Re-run inspection after confirming
the two fields; `ready_for_metric: true` and exit 0 confirm program checks pass.
The explicit CLI equivalents, after confirmation, are `--intrinsics PATH` and
`--intrinsics-shape HEIGHT WIDTH`.

## Convert the original saved predictions

The shared config now includes separate postprocessing paths:

```yaml
input_dir: results/transcg_zero_shot
calibrated_output_dir: results/transcg_metric_calibrated
calibration_root: /public/home/2024141520249/Documents/Projects/ML_pro/data/transcg
zoom_regions: true
```

Run:

```bash
# Optional, no data/model operations: show resolved postprocessing arguments.
python scripts/calibrate_transcg_predictions.py \
    --config configs/transcg_zero_shot.yaml --print-config

# CPU-only conversion, evaluation and all figures.
python scripts/calibrate_transcg_predictions.py \
    --config configs/transcg_zero_shot.yaml
```

CLI settings override the config, for example:

```bash
python scripts/calibrate_transcg_predictions.py \
    --config configs/transcg_zero_shot.yaml \
    --input-dir results/transcg_zero_shot \
    --output-dir results/transcg_metric_calibrated_v2 \
    --depth-max 2 --error-max 1 --no-zoom-regions
```

The CPU command uses `calibrated_output_dir` from YAML; YAML `output_dir` is
still exclusively the inference/visualization path. CPU CLI `--output-dir` and
`--calibrated-output-dir` are aliases. Sampling/inference settings such as
`seed`, `num_samples`, `process_res`, `precision`, and `resume` in the shared
config do not reconfigure old predictions; the original saved metadata controls
their identity. CPU postprocessing requires a fresh output directory and has
no overwrite/resume mode. A second invocation must choose another directory.
It refuses equal, descendant or ancestor output paths relative to the input.

The source manifest, run and per-sample metadata are checked for the official
model ID/architecture, pinned DA3 commit, local checkpoint hashes, canonical-only
units, resize-only single-image inference, RGB identity/hash/actual dimensions,
array shape/dtype/hash and absence of previous metric conversion. Missing or
inconsistent provenance fails rather than being filled with assumptions.
The `checkpoint` supplied by the shared YAML is additionally checked by
streaming SHA256 of its local files, and reading its small architecture JSON;
weights are never deserialized or placed on a GPU. When running without a
`--checkpoint` argument/config field, only saved checkpoint metadata/hash
consistency is checked and `local_files_verified: false` explicitly records
that limitation. Missing/corrupt individual predictions get failure records and
contact-sheet slots; the saved selection is never replaced. These 20-image
diagnostics do not represent the official held-out TransCG test benchmark.

## Exact scale and evaluation protocol

The existing `calibrated_prediction()` calls
`metric_depth_from_canonical()` exactly once, before GT is read:

```text
fx_native = fx_rgb * W_native / W_rgb
fy_native = fy_rgb * H_native / H_rgb
focal_native = (fx_native + fy_native) / 2
metric_depth = canonical_depth * focal_native / 300
```

`H_native, W_native` come from the checked saved array; no nominal process-res
grid is substituted. The pinned [official input processor](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/utils/io/input_processor.py)
scales K rows at boundary resize and again at patch-divisibility resize. Their
composition equals the direct ratios above for the audited single-image
`upper_bound_resize` run. There is no crop or multi-image batch unification.
The [official scaling function](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/utils/alignment.py)
uses mean native focal / 300. No new DA3 preprocessing is executed on CPU.

Only `depth1-gt.png` is Refined GT; `depth1.png` is never substituted or read
for scoring. [Official TransCG conversion](https://github.com/Galaxies99/TransCG/blob/main/utils/data_preparation.py)
divides D435 by 1000 and L515 by 4000 to obtain metres. This CPU command accepts
only D435/camera1. GT must have a checked 16-bit PNG encoding and match RGB;
GT is never resized, filled, normalized or statistically filtered.

The [official data definition](https://graspnet.net/transcg) labels
`depth1-gt-mask.png` as the transparent-object mask. Its nonzero foreground
also supplies the transparent loss region in the
[official reader](https://github.com/Galaxies99/TransCG/blob/main/utils/data_preparation.py).
This is distinct from positive finite GT validity. The local real sample
inspected during development had uint16 GT, uint8 0/1 mask and RGB-matching
dimensions; the server inspector reports the actual selected files independently.
Unknown encodings or unavailable masks disable region scores for that image,
with a reason. The three regions are:

| Region | Definition before GT/prediction validity intersection |
|---|---|
| Full Image | All RGB pixels |
| Transparent Objects | Nonzero transparent-object mask |
| Non-transparent Regions | Mask complement, including ordinary objects/background |

AbsRel, RMSE and delta1 use positive finite GT and positive finite prediction
overlap. Predictions alone are bilinearly resized to the RGB/GT grid using the
existing invalid-support-preserving helper. Coverage and valid GT counts are
reported per image/region. Aggregation pools pixels; RMSE uses pooled squared
errors. Mask-free images contribute only to full-image scores; region sample
counts identify that difference. Empty regions have unavailable/null scores.

## Outputs and audit of no GT alignment

```text
results/transcg_metric_calibrated/
  sample_manifest.json             # byte copy of original, same order/selection
  source_run_metadata.json         # original metadata copy
  run_metadata.json                # source/calibration hashes, CPU-only flags
  inference_summary.json           # operation explicitly CPU calibration
  inference_failures.json
  predictions/sceneN/<same stable hashed ID>/
    canonical_depth.npy            # byte copy, original canonical SHA256 retained
    source_prediction.json         # byte copy of original prediction metadata
    metric_depth.npy               # native grid metres, unclipped
    prediction.json                # source hashes, K/native K, factor, units, scores
    predicted_depth.png
    absolute_error.png              # only when valid GT encoding exists
    visualization.json
    failure.json                   # if this selection failed validation
  metrics_summary.json             # raw scores for three regions
  metrics_summary.csv
  evaluation_status.json           # instead of scores if no valid GT/pred overlap
  figures/sample_001.png            # RGB | GT | metric | absolute error | mask
  figures/sample_001_transparent_zoom.png  # optional mask-bounding-box zoom
  contact_sheet.png                # every original selection, including failures
  visualization_summary.json
```

GT/prediction use the same viridis metre range. Error uses magma in metres;
invalid overlap is grey. Display limits never clip saved arrays. Optional zooms
use only the mask bounding box, not GT error/scale statistics. Canonical copies
are independent files rather than hardlinks, so later derived-file changes
cannot mutate the source. Inference time/memory fields are explicitly copied
from the old run, with a separate CPU postprocessing duration.

Audit `prediction.json`: `metric_conversion.gt_used` is false,
`applied_exactly_once` is true, `gt_alignment` is false, and the multiplier must
equal `focal_native_px / 300`. Recomputing
`canonical_depth.npy * multiplier` must reproduce `metric_depth.npy` within
float32 rounding, independent of GT. Source canonical/prediction/config and K
SHA256 link every derived sample to its originals. The tests change GT while
asserting the metric array stays identical and only scores change.

To redraw only the new results on CPU:

```bash
python scripts/visualize_transcg.py \
    --config configs/transcg_zero_shot.yaml \
    --output-dir results/transcg_metric_calibrated
```

Use the output override here: the general visualization command still reads
YAML `output_dir`, which intentionally points to the original inference run.

## Development verification

```bash
python -m pytest -q --basetemp .test-tmp-transcg-calibration
python -m compileall -q src scripts tests
```

Synthetic tests cover matrix/pose rejection, camera mismatch, discovery ambiguity,
actual native-grid scaling, repeat-conversion refusal, GT-unit/invalid-pixel and
region handling, source immutability, corrupted provenance, config overrides,
no GT-derived scale and a full CPU CLI run with torch/DA3/HF imports forbidden.
Official-processor tests may skip when official DA3 source is unavailable; no
dependency is installed for them. Real server calibration, its camera/grid
provenance, saved-source compatibility, region alignment and numerical/visual
results still require the user's server execution. No real GPU inference or
real TransCG calibration/evaluation was performed in development.
