# ClearGrasp paired transparent–opaque experiment

This controlled appearance comparison reuses the **original 20 real-test sample
IDs and saved transparent metric predictions**. It makes no claim that transparent
inputs will perform worse. No actual GPU experiment or real paired scores were
produced during implementation.

## Server execution

Edit `configs/cleargrasp_paired.yaml` first. Set `data_root` to the actual absolute
Linux directory containing `real-test/`; verify `transparent_output` points to the
completed original 20-image run, and `checkpoint` to that run's official local
DA3METRIC-LARGE directory. Paths follow existing ClearGrasp flat YAML conventions:
relative paths resolve against the repository, environment-variable paths are
rejected, and paths cannot be overridden by command-line flags. The default
`data_root: null` deliberately prevents guessing a server path.

```bash
conda activate vggtomast3r
cd ~/Documents/Projects/ML_pro/ML_project
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# No dataset/model access; inspect resolved configuration.
python scripts/run_cleargrasp_paired.py --config configs/cleargrasp_paired.yaml --print-config

# CPU preflight: source manifest, all 20 transparent files, checkpoint hashes,
# camera K, opaque RGB, GT and mask. No writes and no model initialization.
python scripts/run_cleargrasp_paired.py --config configs/cleargrasp_paired.yaml --dry-run

# Infer ONLY corresponding opaque RGB, one image at a time, one model instance.
python scripts/run_cleargrasp_paired.py --config configs/cleargrasp_paired.yaml

# Continue an interrupted run; verify completed predictions before skipping them.
python scripts/run_cleargrasp_paired.py --config configs/cleargrasp_paired.yaml --resume

# CPU-only analysis and fixed-range PNG generation are explicit separate steps.
python scripts/evaluate_cleargrasp_paired.py --config configs/cleargrasp_paired.yaml
python scripts/visualize_cleargrasp_paired.py --config configs/cleargrasp_paired.yaml

# Reuse an already completed identical analysis/render after checksum validation.
python scripts/evaluate_cleargrasp_paired.py --config configs/cleargrasp_paired.yaml --resume
python scripts/visualize_cleargrasp_paired.py --config configs/cleargrasp_paired.yaml --resume
```

All three entrypoints support `--config`, `--print-config`, `--dry-run`, and
`--resume`. Analysis/visualization dry runs validate saved paired predictions
without writes. Inference does not launch analysis automatically. Commands return
nonzero when any original sample is excluded or a stage fails. A missing pair is
reported, never replaced; other valid original pairs may still run. Do not ignore
nonzero exits or assume 20 successful pairs from an existing directory alone.

Real inference requires the existing server environment, RTX 4090D, `cuda:0`,
BF16, `process_res=560`, offline local checkpoint and unchanged official weights.
The original transparent run must match the same checkpoint byte hashes, pinned
DA3 code commit, model, precision, device, resize method, resolution and offline
RGB-only protocol. A mismatch aborts before model loading. No environment is
created or upgraded by this workflow.

## Manifest, calibration and immutable provenance

The source `sample_manifest.json` is the sole sample selection; no discovery or
random image selection occurs. It must contain exactly 20 unique real-test IDs,
and match the source run metadata's ordered selection. Camera, split, sample ID
and annotation filename must agree. For `real-test/d415/000000029`, matching uses
`000000029-opaque-rgb-img.jpg` in that exact camera/split directory. It never
matches by sorted-list position or across cameras.

`load_result_manifest`, `resolve_prediction_dir` and strict
`saved_prediction(..., require_metric=True)` reuse the project's committed
ClearGrasp migration interface. It supports readable and old 24-character hash
folders, including old relative/split-prefixed ID aliases, and rejects ambiguous
duplicates. Shared Booster/TransCG paths are untouched.

Transparent `metric_depth.npy` is read only. Its `prediction.json` identity,
output hashes, explicit metre units, trusted single focal conversion and native
dimensions are checked. Current transparent RGB and calibration must match saved
hashes and K. Current GT/mask hashes are pinned in the paired manifest. Analysis
rechecks inputs, original manifests/metadata, saved prediction bytes and camera
calibration; it never rescales the saved metric array a second time.

Opaque inference reuses `DA3Predictor.predict_canonical()` and
`metric_depth_from_canonical()`, using the same resize-only preprocessing and K.
RGB resolutions and native output grids must match. Conversion is exactly
`D_m = D_canonical * mean(fx_native, fy_native) / 300`. GT never enters inference,
model input, conversion, or raw metrics. There is no automatic registration,
cropping, GT resizing, mask resizing, or nonrigid warp.

New output and source results, dataset and checkpoint are kept disjoint. New
opaque files have explicit `opaque_` prefixes under
`predictions/real-test/<camera>/<number>/`. Transparent paths remain in their
original run and are referenced with checksums in `paired_manifest.json`.
Opaque folders are assembled in `.staging/` and renamed only when complete, so
interrupted staging directories can remain untouched while `--resume` continues.
Existing prediction files are never overwritten. Unexpected partial files in a
final sample folder are reported instead of silently adopted.

## GT, registration and scene review

The [official ClearGrasp evaluator](https://github.com/Shreeyak/cleargrasp/blob/master/eval_depth_completion/eval_depth_completion.py)
distinguishes real transparent sensor depth from opaque GT depth and uses nonzero
object masks. This experiment follows that opaque GT convention. Dataset capture
provenance and equal dimensions alone do not certify per-pixel registration of
each RGB pair. Use the paired RGB/GT/mask panels for review, including object
edges and stationary background features. Raw-region and geometry comparisons
remain subject to residual motion/lighting differences even when pixelwise
differences are disabled.

By default `registration_review: null`: direct prediction/log-depth differences
are skipped, with explicit status/reason and a skipped panel. No hidden geometric
correction is performed. To enable direct differences, create a JSON review
outside output directories and set its path in YAML **before inference**:

```json
{
  "real-test/d415/000000029": {
    "verified": true,
    "reviewer": "reviewer name",
    "reason": "Record the RGB/GT/mask alignment evidence and capture/scene source here",
    "scene": "verified_capture_scene_id",
    "transparent_sha256": "actual 64-character RGB SHA-256",
    "opaque_sha256": "actual 64-character RGB SHA-256",
    "gt_sha256": "actual 64-character GT SHA-256",
    "mask_sha256": "actual 64-character mask SHA-256"
  }
}
```

The example is a schema, not an approval or evidence. Checksums must be actual
file digests. Run the CPU inference dry run to inspect the paired records and
checksums without producing output. Review entries may use `verified: false`
with `reason` explaining rejection. A known scene can still be recorded for a
rejected/unverified pair, but requires reviewer and source evidence. Every entry
must reference one of the original IDs. Scene identifiers are manual, evidence
based, and never inferred from neighboring frame numbers. Camera is recorded
separately and is **not** treated as a scene. Missing scene information stays
unknown; the effective independent scene count is reported as unavailable if
any evaluated sample has unknown membership.

An existing paired run pins the review records. To change reviews, grouping,
inference or analysis configuration, choose a new paired output directory (or
finish review before starting). `--resume` accepts identical provenance and
verified artifacts, not an instruction to overwrite a previous experiment.

## Evaluation definitions

For each pair use **the same** intersection `V = valid(GT) & valid(T) & valid(O)`,
where validity means finite positive values. Predictions alone are bilinearly
resized to the unchanged GT grid with the existing invalid-support policy.
The object mask, non-object region and valid GT have separate meanings.

1. **Raw metric:** AbsRel, RMSE and delta1 for full image, object and background.
   No GT alignment. Official paired tables use the shared support only. There
   are no independently-valid-pixel scores mixed into these tables.
2. **Scale:** per-region median pixelwise pred/GT ratio and median natural-log
   pred/GT residual. `paired_differences.csv` stores T-O differences, the T/O ratio
   of median pred/GT ratios, and the log bias difference. A positive signed bias
   shift is not automatically worse accuracy.
3. **Background aligned diagnostic:** each input separately uses
   `s = median_BG(GT/pred)` over the same shared background. Report object and
   background AbsRel after scaling, separately from raw metres. Every diagnostic
   carries **GT-dependent diagnostic only — not deployable metric performance.**
4. **Material relative scale:**
   `delta_material = (median_obj log(T/GT) - median_BG log(T/GT))`
   `                 - (median_obj log(O/GT) - median_BG log(O/GT))`.
   Near zero means no extra relative object/background scale contrast. Positive
   means the transparent object's relative depth is larger; negative means
   smaller, compared with opaque. The statistic is invariant to independently
   multiplying each whole prediction by any positive constant. Neither sign
   establishes the worse geometry without the other metrics. Report supporting
   object/background pixels and the number of contributing sample pairs.
5. **Relative geometry:** reuse existing relative geometry, oracle scale and
   SILog definitions. Object/object, local object/background, background/background
   and full ordinal comparisons draw/accept pixel pairs once from common
   support and GT with the same seed. Both predictions score these exact arrays.
   The existing penetration calculation consumes the same supplied local
   candidates and GT acceptance to measure reversal and log-gap compression.
   Ordinal and penetration use original unaligned predictions. The opaque input
   uses the same physical-object mask as the transparent input. Minimum unique
   pair and endpoint thresholds remain explicit, and unsupported scores are null.
6. **Reviewed pixel differences:** raw `T-O`, `log(T)-log(O)` and the log map minus
   its shared-background median, with object/background/boundary median summaries.
   These are generated only for approved, checksum-bound registration reviews.

The original-mask pass is always preserved. A second pass erodes object and
background separately by `erosion_margin` (default 2 pixels), excluding an
uncertain boundary band rather than assigning removed object pixels to background.
Original/eroded rows report sensitivity for raw, scale, material and geometry.
GT alignment never modifies stored predictions.

`paired_summary.json` reports per-metric contributing pair count, equal-pair mean
and median T-O difference, transparent-worse/opaque-worse/tie counts for errors
and accuracy, and percentile 95% bootstrap CI of the **mean paired difference**.
Accuracy/delta1 use the correct opposite direction for worse counts. Signed
scale/material shifts instead report positive/negative counts. Bootstrap uses a
fixed seed, default 10,000 resamples of whole sample pairs, with no favorable-seed
search or outcome-based exclusion. For complete scene labels and at least two
verified camera/scene groups it also reports equal-scene mean differences and
scene-cluster bootstrap CIs by resampling whole groups. With missing groups the
scene CI is unavailable. Frame bootstrap intervals are descriptive and can
underestimate uncertainty from correlated captures. Small group counts still
limit inference; do not call 20 frames 20 independent scenes.

## Outputs

`results/cleargrasp_paired/` contains the immutable paired manifest, run metadata,
inference summary/failures and opaque-only prediction folders. The metrics folder
contains `raw_metric_comparison.csv`, `scale_comparison.csv`,
`relative_geometry_comparison.csv`, `material_scale_bias.csv`,
`paired_differences.csv`, `paired_summary.json`, `evaluation_failures.json` and
`report.md`. CSVs contain every contributing pair's original values, camera,
known scene/group, support and sensitivity pass; missing values are blank/null,
never invented zeroes. Manifest and failure files retain excluded original IDs.

Each `figures/real-test/<camera>/<number>/` contains RGB, GT/mask/raw depth, raw
error, background aligned diagnostic, sampled ordinal endpoint error fraction,
and reviewed/skipped prediction difference PNGs, plus visualization metadata.
Depth panels share the configured `[0, depth_max]` metre limits for GT and both
raw predictions across all images; error panels use `[0, error_max]`. Signed
differences use symmetric fixed limits. There is no per-image percentile
normalization. Increase a configured range in a **new** output if saturation
obscures structure. `contact_sheet.png` contains all successfully rendered pairs;
the rendering summary lists selected/rendered/skipped/failures. Contact sheet
tiles carry camera/sample IDs, GT, object mask, transparent and opaque depth.

Analysis and visualization refuse existing outputs by default. `--resume`
revalidates inputs, exact settings and all generated file hashes. If a previous
analysis/render itself was interrupted before its artifact metadata was complete,
the incomplete directory is preserved and a new output is required; prediction
inference can still resume safely. This deliberately avoids silent replacement
of analysis artifacts. Inference and evaluation failures are always visible;
no command silently upgrades partial completion to success.

## CPU validation

```bash
python -m pytest tests/test_cleargrasp_paired.py tests/test_cleargrasp.py \
    tests/test_relative_geometry.py tests/test_cleargrasp_geometry_analysis.py -q
python -m pytest -q
```

Synthetic tests mock the predictor and EXR reader, verify one opaque-only model
load, original-manifest-only selection, both legacy/readable paths, calibration,
unchanged transparent bytes, no-overwrite/resume, shared support and exact ordinal
endpoints, material scaling invariance, geometry degradation, registration gates,
scene/bootstrap behavior, and rendered PNG output. They do not establish real
checkpoint loading, EXR codec availability on the server, real pairing quality,
runtime performance or experimental metric values. Run the server preflight and
review actual captures before drawing conclusions.
