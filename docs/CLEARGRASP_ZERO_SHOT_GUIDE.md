# ClearGrasp DA3METRIC-LARGE zero-shot development and server guide

This implementation asks whether DA3 preserves transparent physical surface
geometry and foreground/background order. Raw metric error alone cannot answer
that question. **No real DA3 inference was executed during Windows development.**
No weights were downloaded, no environment was created or upgraded, and original
ClearGrasp, Booster and TransCG data/predictions were not changed.

## Protocol references

- [Official ClearGrasp evaluation](https://github.com/Shreeyak/cleargrasp/blob/master/eval_depth_completion/eval_depth_completion.py):
  distinguishes opaque GT from transparent sensor depth; uses nonzero segmentation
  masks and finite positive GT. This implementation pairs by ID instead of sorting
  separate file lists, and resizes predictions only.
- [Official EXR utilities](https://github.com/Shreeyak/cleargrasp/blob/master/api/utils.py):
  scalar R channel, floating metre depth, zero denotes holes.
- [Official mask loader](https://github.com/Shreeyak/cleargrasp/blob/master/pytorch_networks/masks/dataloader.py):
  synthetic object IDs start at 100. Local distributed segmentation PNGs were
  independently inspected and all use binary 0/255.
- [Dataset page](https://sites.google.com/view/cleargrasp/data).

Real GT is the opaque capture paired to the transparent RGB, not transparent
sensor depth or the provided output-depth reconstruction. The official evaluation
uses these pairs in the same pixel frame. We check dimensions and camera identity;
this does not prove subpixel physical registration or eliminate capture motion.
Inspect RGB/mask/GT overlays on the server before interpreting boundary errors.
The optional two-pixel erosion used for geometry pair sampling is explicit and
does not modify annotations or raw region metrics.

## Actual local inventory (2026-10-26)

Root: `E:\cleargrasp-dataset-test-val\cleargrasp-dataset-test-val`.

| Split | Samples | Groups | RGB/GT size (H x W) |
|---|---:|---|---|
| real-test | 113 | d415: 90; d435: 23 | 480 x 848; 720 x 1280 |
| real-val | 173 | d435: 173 | 720 x 1280 |
| synthetic-test | 408 | glass-round-potion-test: 109; glass-square-potion-test: 94; star-bath-bomb-test: 98; tree-bath-bomb-test: 107 | 576 x 1024 |
| synthetic-val | 532 | cup-with-waves-val: 100; flower-bath-bomb-val: 100; heart-bath-bomb-val: 100; square-plastic-bottle-val: 115; stemless-plastic-champagne-glass-val: 117 | 576 x 1024; 1080 x 1920 |

All 1,226 RGB/GT/mask triplets were inspected with the existing repository
`.venv` using OpenCV EXR decoding. Missing required GT/masks: 0; orphan annotations:
0; unreadable files: 0; dimension mismatches: 0. RGB is JPG uint8, GT is EXR decoded
as float32, masks are PNG uint8 with values 0/255. Finite-positive GT fractions:
real-test 0.682852 to 0.936739; real-val 0.496245 to 0.998245; synthetic splits 1.0.
No minimum/maximum depth clipping or inpainting is applied in evaluation.

Detailed evidence is generated in `results/cleargrasp_local_inspection/`:
`dataset_inspection.json` (all file checks, shapes, mask values and GT coverage),
`sample_manifest.json` (all ID pairs and relative source paths).
Results are gitignored; these local reports are not model results.

Real files are colocated under `<split>/<camera>/`. Synthetic files are under
`<split>/<object-group>/{rgb-imgs,depth-imgs-rectified,segmentation-masks,json-files}`.
Sample identity includes group/camera and numeric frame ID, preventing collisions.
Discovery also accepts one unambiguous wrapper directory above the split root.
It records absent splits instead of manufacturing them. GT/mask orphans and missing
files are reported. Inference can still save RGB predictions if annotations are
missing; evaluation marks the affected sample as failed rather than inventing GT.

## Input, GT and calibration

- Real model input: `*-transparent-rgb-img.jpg` only.
- Real GT: `*-opaque-depth-img.exr`; sensor measurement
  `*-transparent-depth-img.exr` is inspected separately and never passed to DA3.
- Synthetic input: `rgb-imgs/*-rgb.jpg`; GT:
  `depth-imgs-rectified/*-depth-rectified.exr`; mask:
  `segmentation-masks/*-segmentation-mask.png`. Radial/unrectified depth is not used.
- Scalar mask is nonzero = transparent, zero = non-transparent; never GT validity.
- Valid evaluation pixels are finite positive prediction AND finite positive GT,
  intersected with the requested region. No GT/mask resizing or silent channel
  averaging is allowed. Predictions alone use the existing invalid-aware bilinear
  resize to the GT grid.

Each real camera directory has its own `camera_intrinsics.yaml`:

| Calibration | Native dimensions (H x W) | fx / fy | cx / cy |
|---|---|---|---|
| real-test/d415 | 480 x 848 | 616 / 615 | 427 / 245 |
| real-test/d435 | 720 x 1280 | 921 / 921 | 642 / 359 |
| real-val/d435 | 720 x 1280 | 921 / 921 | 642 / 359 |

The reader validates camera name against directory, resolution, finite positive
focal lengths and principal point bounds. Missing/untrusted calibration produces
canonical-only predictions with a warning. **Raw Metric scores then remain
unavailable**, while oracle/log/ordinal diagnostics can use canonical depth.
Synthetic per-image JSON camera/FOV metadata is reported but is not promoted to a
verified K: its rectification/calibration needs separate verification before metric
evaluation. Camera poses/extrinsics are never fed into the model or used for scale.

The existing `metric_depth_from_canonical()` is reused unchanged. For actual DA3
output dimensions `(Hn,Wn)` and RGB `(H,W)`:

```text
Kn = diag(Wn/W, Hn/H, 1) @ K
f_native = (Kn[0,0] + Kn[1,1]) / 2
metric_depth = canonical_depth * f_native / 300
```

This conversion happens once on the network output grid before resizing a copy
for evaluation. Native canonical arrays are retained. Prediction metadata records
original K, output K, actual focal, multiplier, shapes, YAML SHA256 and the
`applied_exactly_once` flag. No GT participates in inference or metric conversion.

## Relative geometry definitions

All thresholds are fixed CLI options and saved with results. Defaults:
seed 42; 20,000 candidate pair draws per image/comparison; GT gap strictly greater
than 0.02 m; local radius 30 pixels; boundary margin 2 pixels; compression ratio 0.25.
Spatial parameters refer to original RGB/GT pixels; datasets with different
resolutions should be interpreted with that limitation in mind.

1. **Raw**: AbsRel, RMSE in metres and delta1 (<1.25), for full/T/background.
   No GT alignment. Dataset aggregation pools pixel sufficient statistics through
   the existing `MetricAccumulator`.
2. **Oracle**: one whole-image factor `median(GT/pred)` over valid pixels.
   This is the median of pixel ratios, not a ratio of medians. A second factor
   uses only valid background GT, then measures residual errors in every region.
   Both are **GT-dependent oracle diagnostic, not deployable metric performance.**
   Factors and aligned errors are recorded separately and never overwrite raw NPYs.
3. **Log**: residual `e=ln(pred)-ln(GT)`; region median bias; SILog =
   `sqrt(mean((e-mean(e))**2))` (natural log, no factor 100); dispersion =
   `median(abs(e-median(e)))`. Diagnostics aggregate as an equal-image mean,
   with unavailable regions excluded and contributing sample counts reported.
4. **Ordinal**: full/full, T/T, T/background, background/background comparisons.
   GT must exceed the configured absolute metre gap; prediction ties count as
   incorrect on these non-tied GT pairs. No prediction scaling occurs. T/T, B/B
   and full/full use uniform sampling with replacement. Local T/background draws
   start in the eroded transparent region and sample offsets in a bounded disk;
   the target must be valid background beyond the boundary margin and inside the
   ring. There is no fallback to arbitrary distant background. Pair counts can be
   less than candidate draws. Sampling is O(pixels + draws), never O(pixels^2).
   Each image seed derives from SHA256(base seed, split, sample ID), so image order
   and partial reruns do not change its pairs. Duplicate Monte Carlo pairs may
   occur and counts must not be treated as independent statistical observations.
5. **Penetration**: retain local pairs whose GT verifies T is in front of B by
   more than the gap. A negative `ln(pred_B/pred_T)` is reversal; a nonnegative gap
   below `0.25 * ln(GT_B/GT_T)` is severe compression. These disjoint errors are
   reported separately and as their union, with denominators. Log gaps make this
   invariant to global scale. If no reliable local front pairs exist, the result
   is `unavailable`. A nearby background comparison cannot prove the exact unseen
   surface behind a transparent pixel; it diagnoses local ordering/compression,
   not transmission or optical path reconstruction.

Ordinal and penetration dataset summaries pool accepted sampled-pair counts, and
per-image counts/results remain available. Region ordinal table columns mean
full/full, T/T, B/B respectively. T/background is reported separately.
The summary table is generated only from actual saved arrays. It is explicitly
the project's protocol, not an official ClearGrasp benchmark. Val/test and
real/synthetic are selected separately; nothing trains on these splits.
Optional boundary error metrics/crops/ordinal error overlays are not implemented.

## Windows inspection and development checks

Use the existing environment; do not install over the server environment:

```powershell
cd D:\junior\Intro_ML\Projects\ML_project
.venv\Scripts\python.exe scripts/inspect_cleargrasp.py `
  --data-root 'E:\cleargrasp-dataset-test-val\cleargrasp-dataset-test-val' `
  --output-dir results/cleargrasp_local_inspection

.venv\Scripts\python.exe -m pytest -q --basetemp .test-tmp-cleargrasp-final
```

EXR backends: existing OpenEXR+Imath first, then existing OpenCV with
`OPENCV_IO_ENABLE_OPENEXR=1`. OpenCV BGR selects R, not B. No imageio download or
package installation occurs. The system Python initially lacked all EXR backends;
the existing `.venv` was subsequently confirmed to decode all local EXRs.
If decoding is unavailable on Linux, inspection still writes JSON with individual
errors and exits nonzero. Inspect imports/build support in the existing environment
and resolve dependencies explicitly; no Torch upgrade or new Conda environment.
NumPy, Pillow, PyYAML, SciPy and Matplotlib are also needed; the existing repository
dependency reference is not an instruction to reinstall the environment.

Tests use small generated images/depth arrays and a fake predictor, including
offline/resume behavior and visualization. Fake outputs are only test fixtures.
They are not ClearGrasp inference results.

Development verification on 2026-10-26: the existing `.venv` full suite completed
with **113 passed**, plus 16 Matplotlib/PyParsing deprecation warnings. New ClearGrasp
and relative geometry tests account for 20 cases. Python compile checks and CLI
help/config parsing succeeded. Synthetic seven-panel figures were rendered and
visually checked. A separate read-only three-row real annotation preview is saved
as `results/cleargrasp_local_inspection/dataset_registration_preview.png`; it contains
RGB, nonzero-mask overlays and opaque GT only, with no model prediction. The sampled
overlays cover the transparent objects; GT holes remain visible. This is not a
claim of perfect registration across the entire dataset. The Linux Bash wrapper
and real GPU inference have not been executed locally.

## Linux smoke test

Use the existing server environment, official DA3 source commit
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` and local checkpoint directory. The
existing predictor validates source provenance, architecture, runtime versions,
precision and RTX 4090D. One lazily initialized predictor handles independent RGB
calls; metadata preserves timing/peak allocated VRAM per image. No training.

```bash
conda activate vggtomast3r
cd ~/Documents/Projects/ML_pro/ML_project
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CLEARGRASP_ROOT=/path/to/cleargrasp-dataset-test-val

python scripts/inspect_cleargrasp.py \
  --data-root "$CLEARGRASP_ROOT" \
  --output-dir results/cleargrasp_smoke

python scripts/run_cleargrasp_zero_shot.py \
  --data-root "$CLEARGRASP_ROOT" \
  --checkpoint checkpoints/DA3METRIC-LARGE \
  --split real-test --max-samples 20 --seed 42 \
  --process-res 560 --device cuda:0 --precision bf16 \
  --output-dir results/cleargrasp_smoke
```

Or after inspection: `bash scripts/run_cleargrasp_zero_shot.sh` (same smoke defaults;
extra CLI arguments are forwarded). Use a new output directory if changing inputs,
selection, checkpoint bytes, process resolution, device or precision. Resume is
automatic for exactly matching configurations with valid output checksums. Partial
NPYs lacking `prediction.json` are regenerated. Completed but corrupt outputs are
reported as failures; do not silently trust them. The model is not loaded if every
selected prediction resumes. Per-run configuration binds input RGB and calibrated
K hashes, and evaluation stores GT/mask hashes. Outputs are atomic where shared
helpers support them. Concurrent writers to one output directory are not supported.

After smoke completion, inspect `inference_failures.json`,
`metrics/evaluation_failures.json`, timing/VRAM and figures. Verify calibration
metadata, GT registration, fixed depth colour ranges and local pair coverage.
Colour saturation above a chosen maximum is explicitly shown by extended colourbars.
GT-invalid locations are grey in GT/prediction/error depth panels. All samples use
the same configured metre ranges; missing metric calibration is visibly unavailable.
Aligned panels are labeled diagnostic only. Seven individual panels plus one
comparison image are saved per image, and `contact_sheet.png` covers the run.

Only after reviewing a successful smoke test, run the complete real test split:

```bash
python scripts/inspect_cleargrasp.py \
  --data-root "$CLEARGRASP_ROOT" --output-dir results/cleargrasp_real_test

python scripts/run_cleargrasp_zero_shot.py \
  --data-root "$CLEARGRASP_ROOT" \
  --checkpoint checkpoints/DA3METRIC-LARGE \
  --split real-test --all --seed 42 \
  --process-res 560 --device cuda:0 --precision bf16 \
  --output-dir results/cleargrasp_real_test
```

`--all` and `--max-samples` are mutually exclusive. The default is 20 samples;
smaller splits use all available samples without replacement. All four discovered
split names are supported. Standalone CPU reevaluation/replotting needs no model:

```bash
python scripts/evaluate_cleargrasp.py \
  --output-dir results/cleargrasp_smoke --data-root "$CLEARGRASP_ROOT" \
  --seed 42 --pair-draws 20000 --gt-gap-m 0.02 \
  --ring-radius 30 --boundary-margin 2 --compression-ratio 0.25

python scripts/visualize_cleargrasp.py \
  --output-dir results/cleargrasp_smoke --data-root "$CLEARGRASP_ROOT" \
  --depth-max 2.0 --error-max 0.5
```

The YAML file is a documented reference matching CLI defaults, not an implicitly
loaded config. CLI flags are authoritative. `--print-config` prints parsed runner
options without touching data/checkpoints or importing Torch.

## Output and remaining validation

Outputs include inspection JSON, selected manifest, run metadata, inference summary
and failures; `predictions/<split>/<stable-ID-hash>/` contains original canonical
NPY, optional metric NPY, prediction metadata. Hash-based paths are collision
resistant; the original sample ID is stored in both manifest and prediction JSON.

`metrics/` contains raw JSON/CSV, relative summary JSON, relative per-image JSON/CSV,
scale diagnostics CSV, evaluation failures and `summary.md`. Figures have seven
standalone PNGs plus comparison PNG per sample; the root has a contact sheet and
visualization summary. JSON uses null/unavailable rather than NaN or invented data.

Still requiring server verification: real checkpoint loading with the pinned
official source; BF16 single-image output and actual native dimensions; CUDA timing
and peak memory; focal-to-metre behavior on real images; capture registration and
interpretation of real local background pairs; smoke and full test inference.
No real model metric scores are claimed by this development delivery.
