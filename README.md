# DA3METRIC-LARGE zero-shot on processed Booster

The sole model is **`depth-anything/DA3METRIC-LARGE`**. Final confirmed settings:
**RTX 4090D**, `process_res=560`, BF16 preferred (`--precision auto`), one RGB
image per call. The existing server Conda environment is **`vggtomast3r`**.
No training is implemented. Future Head-only and LoRA + Head work will use an
A800 80GB under a combined 60-hour budget; it is outside this change.

**Raw Metric Evaluation uses no GT scale/shift alignment.** Standalone Metric-Large
network output is not assumed to be metres: calibrated native focal conversion
is mandatory. Old BASE/SMALL predictions are rejected by this evaluator.

## What has actually been tested

See [VALIDATION.md](docs/VALIDATION.md) and
[metric_large_validation.json](docs/metric_large_validation.json).

- The G drive was remounted and all 76 samples were re-audited successfully.
- Final user-confirmed input is **448x560 (H,W)**, achieved with process_res=560.
  Official preprocessing/K scaling are checked for all 76 samples.
- The conversion matches official `apply_metric_scaling` in numerical tests.
- CPU/unit/integration tests pass without downloading weights.
- The server forbids SSH. No 4090D execution or real Metric-Large checkpoint
  inference has been performed from this session; **no Metric-Large experiment
  score is claimed**. Previous BASE scores remain historical evidence only.

## Reuse the existing server environment

Do not run `conda create`, create a venv, upgrade torch, or install the old
requirements automatically. `environment.yml` has been removed. The retained
`setup_env.sh` is now a **read-only preflight**, with no install/clone/patch steps.
`requirements.txt` lists a previously tested dependency reference, not an
instruction to replace working dependencies in the research environment.

From a server terminal or its permitted web terminal:

```bash
# Use a new checkout, or git pull --ff-only in a clean existing zero_shot checkout.
git clone --branch zero_shot https://github.com/C2H5O/ML_project.git
cd ML_project

conda run --no-capture-output -n vggtomast3r python scripts/check_environment.py \
  --strict --import-da3 --device cuda:0 --require-gpu 'RTX 4090D' \
  --output logs/metric_large_environment.json
# Equivalent read-only preflight:
# bash setup_env.sh
```

The check records actual Python/PyTorch/CUDA/GPU/GLIBC/GCC/dependency/source
versions. Compatibility bounds preserve the stated server limits (Python >=3.9
and <=3.10.20; torch >=2.0 and <=2.3.1), without requiring a reinstall to one
exact version. Imported torchvision and an actual CUDA operation are checked.
The selected device must be RTX 4090D; an RTX 4090 or another GPU does not pass.
BF16 is preferred; unsupported explicit BF16 requests fail, while auto can choose
FP16. The tool does not mutate the environment when a check fails.

The supported official DA3 source commit is
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. If the existing environment imports a
different source or has missing dependencies, retain the diagnostic and resolve
that specific discrepancy before inference; do not run a blanket installer.
If the verified source already exists outside site-packages, expose that source
for this process with `PYTHONPATH=/path/to/Depth-Anything-3/src:$PYTHONPATH`.
The optional lazy-export patch tool is retained for review, and is never run by
preflight or inference. No network/weight/preprocessing computation is replaced.

## Calibrated metric conversion

Official references:

- [Metric FAQ](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/README.md#-faq)
- [Official scaling function](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/utils/alignment.py)

The saved Booster GT is already in metres. It is not re-converted. Its metadata
contains `K` at the saved 448x560 RGB grid and `K_original` plus resize/crop
geometry. The loader requires those to agree before using calibrated focal.

For resize-only official processing and actual network output Hn x Wn:

```text
K_native = diag(Wn / Wsaved, Hn / Hsaved, 1) @ K_saved
f_native = (K_native[0,0] + K_native[1,1]) / 2
depth_metric_m = canonical_network_output * f_native / 300
```

The scale uses actual output dimensions after all patch rounding, not the
original 3008x4112 camera focal. The final user-confirmed process_res=560 keeps
the saved 448x560 grid unchanged, so K_native=K_saved. The superseded setting
process_res=448 would give 364x448 (sx=.8, sy=.8125); it is covered by tests but
is not the configured baseline.
Metric conversion is performed exactly once **before** bilinear resizing back
to the saved GT grid. No disparity conversion, stereo baseline multiplication,
GT median, GT fitting, predicted camera focal or scene-dependent tuning is used.
The model forward receives RGB only; calibration is used in the explicit output
conversion. Unexpected already-metric upstream outputs fail rather than risk
applying the focal conversion twice.

## Server data and smoke test

The processed Windows data must already be transferred to a server path. Keep
all `train/index.json` and indexed sample directories intact. Neither this code
nor any command below modifies the dataset or regenerates GT.

```bash
export DATA_ROOT=/absolute/server/path/processed/booster

conda run --no-capture-output -n vggtomast3r python scripts/inspect_booster.py \
  --data-root "$DATA_ROOT" --output-dir logs/metric_large_audit

conda run --no-capture-output -n vggtomast3r python scripts/validate_metric_scale.py \
  --data-root "$DATA_ROOT" --output logs/metric_scale_validation.json

# Full smoke pipeline: preflight, 3 predictions, Raw Metric evaluation, 3 figures.
OUTPUT_DIR=results/metric_large_smoke bash scripts/run_zero_shot.sh --max-samples 3
```

By default the official checkpoint is loaded from Hugging Face. Its ref is
resolved to an immutable SHA and stored. To use already available offline weights:

```bash
export CHECKPOINT_DIR=/absolute/server/path/DA3METRIC-LARGE
# This must contain config.json and model.safetensors for model_name=da3metric-large.
OUTPUT_DIR=results/metric_large_smoke bash scripts/run_zero_shot.sh --max-samples 3
```

No credentials or local personal path is committed. Local checkpoint hashes are
recorded. The first online inference may download the specified checkpoint;
preflight does not download anything.

## Full inference, evaluation and figures

After reviewing the smoke files, run the existing train split:

```bash
bash scripts/run_zero_shot.sh
```

The wrapper uses the same existing environment and selection for all stages.
`DEVICE=cuda:1` may select a different visible device, but it must still be a
4090D. `--scene NAME` (repeatable), `--subset balanced|unbalanced`, and
`--max-samples N` can be passed to the wrapper and are forwarded to all stages.
Default results go to `results/zero_shot_metric_large`, preserving older outputs.
For an explicit run without the wrapper:

```bash
conda run --no-capture-output -n vggtomast3r python scripts/run_zero_shot.py \
  --data-root "$DATA_ROOT" --split train \
  --model-id depth-anything/DA3METRIC-LARGE \
  --process-res 560 --device cuda:0 --require-gpu 'RTX 4090D' --precision auto \
  --output-dir results/zero_shot_metric_large --seed 42 --resume

conda run --no-capture-output -n vggtomast3r python scripts/evaluate_zero_shot.py \
  --data-root "$DATA_ROOT" --split train --mode raw_metric \
  --output-dir results/zero_shot_metric_large

conda run --no-capture-output -n vggtomast3r python scripts/visualize_zero_shot.py \
  --data-root "$DATA_ROOT" --split train \
  --output-dir results/zero_shot_metric_large \
  --scene Bathroom --scene Mirror --scene Bottle --max-samples 6 \
  --depth-min 0 --depth-max 5 --error-max 1 --dpi 200
```

There is no automatic model fallback or silent resolution change. OOM reports a
failure; explicitly lower `PROCESS_RES` in a fresh output directory if needed.
No other job is stopped and no shared GPU/environment settings are modified.

## Evaluation and saved results

- AbsRel, RMSE and delta1 use finite positive GT and predictions, intersected
  with saved validity. `--validity consistent` additionally applies the separate
  stereo consistency mask and must match the figure protocol.
- Aggregation is global pixel-weighted, including global squared-error RMSE.
- All material classes 0-3, target 2+3, ordinary 0 and other 0+1 are reported.
  Material severity masks cannot separately identify transparent versus mirror.
- Empty regions are unavailable (`null`), not zero. Counts and prediction
  coverage are saved. Failures produce records and nonzero exit status.
- Only supplied train is processed; these are not held-out test scores. Dataset
  calibration warnings and possible raw split name overlap remain documented in
  [DATASET_STRUCTURE.md](docs/DATASET_STRUCTURE.md).

```text
results/zero_shot_metric_large/
  run_metadata.json             # exact model/source/runtime/precision/protocol
  inference_summary.json
  inference_failures.json
  predictions/<split>/<hashed ID>/
    canonical_native.npy       # untouched official output, not assumed metres
    raw_native.npy             # calibrated metric metres at actual native grid
    raw.npy                    # raw metric metres mapped to saved RGB/GT grid
    prediction.json            # K_saved, K_native, focal, factor, timing, checksums
  metrics_per_image.csv
  metrics_summary.json
  metrics_summary.csv
  failures.json
  figures/*.png                # RGB / GT / raw metric / abs error / rel error / material
  visualization_settings.json
  visualization_failures.json
```

No `scale_aligned.npy` or GT alignment result is produced. Evaluator verifies the
canonical-to-metric conversion and calibration hashes before accepting arrays.
Resume also binds RGB, calibration metadata, config and all three array hashes.
GT and masks are not resized. Figures use common metre ranges for GT/prediction
and preserve every numerical output.

To visualize all 76 existing predictions without running inference again:

```bash
conda run --no-capture-output -n vggtomast3r python scripts/visualize_zero_shot.py \
  --data-root "$DATA_ROOT" --split train \
  --output-dir results/zero_shot_metric_large \
  --depth-min 0 --depth-max 5 --error-max 1 --dpi 200
```

The default is all selected samples, rather than a five-image preview. Set
`--max-samples 76` explicitly if desired. Each PNG contains RGB, GT, raw metric
depth, absolute error, relative error and material overlay. The script also
writes `visualization_manifest.csv` (sample IDs to PNG paths),
`visualization_summary.json` and `visualization_failures.json`. Missing or invalid
predictions are recorded as failures and cause a nonzero exit status.

Historical BASE predictions/results remain distinguishable at their original
locations and commit. They are not results for this Metric-Large protocol. See
[the detailed protocol](docs/ZERO_SHOT_GUIDE.md) for audit limitations and tests.

## TransCG zero-shot diagnostic (independent of Booster)

Already have canonical predictions? Use the **CPU-only calibration workflow**:
[TransCG existing-prediction calibration guide](docs/TRANSCG_CPU_CALIBRATION.md).
Run `scripts/inspect_transcg_calibration.py --config configs/transcg_zero_shot.yaml`
on the server first, then confirm the D435 K path and its original pixel grid.
`scripts/calibrate_transcg_predictions.py --config configs/transcg_zero_shot.yaml`
converts the original manifest's saved arrays into a separate
`results/transcg_metric_calibrated/`, evaluates full/transparent/non-transparent
regions and regenerates metric/error figures without importing torch or DA3.
It never reruns inference or changes the original zero-shot directory.

This workflow is implemented for **DA3METRIC-LARGE** and does not represent a
completed TransCG experiment. It uses only local weights and the existing
`vggtomast3r` environment; it does not install packages, train, download models,
query online HF revisions, or read Booster data. The DA3 source must remain at
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. The existing GPU guard expects an
RTX 4090D. The shared loader verifies the checkpoint architecture and hashes.

The editable configuration is **`configs/transcg_zero_shot.yaml`**. It includes
the supplied server checkpoint path and dataset base directory:

```yaml
data_root: /public/home/2024141520249/Documents/Projects/ML_pro/data
checkpoint: /public/home/2024141520249/Documents/Projects/ML_pro/ML_project/checkpoints/DA3METRIC-LARGE
output_dir: results/transcg_zero_shot
num_samples: 20
seed: 42
process_res: 560
device: cuda:0
precision: bf16
resume: false
```

Edit this file to change paths, scene range, camera, sample count, sampling mode,
seed, resolution, device, precision, calibration and visualization colour limits.
`data_root` must point to the extracted dataset through a supported layout; the
server directory has not been inspected locally. If you extract the archive to
`/public/home/2024141520249/Documents/Projects/ML_pro/data/transcg-data-1/transcg`,
set `data_root` to that exact directory. The supplied base `data` path works if
it directly contains scenes or `transcg/scene1`, for example. The config does
not create/move datasets or download anything.

All three TransCG Python commands accept the same configuration:

```bash
conda activate vggtomast3r
cd /public/home/2024141520249/Documents/Projects/ML_pro/ML_project
export HF_HUB_OFFLINE=1

# Print effective options only: no dataset scanning, torch import or inference.
python scripts/run_transcg_zero_shot.py --config configs/transcg_zero_shot.yaml --print-config

python scripts/inspect_transcg.py --config configs/transcg_zero_shot.yaml
python scripts/run_transcg_zero_shot.py --config configs/transcg_zero_shot.yaml

# Resume with an explicit override, or set resume: true in YAML.
python scripts/run_transcg_zero_shot.py --config configs/transcg_zero_shot.yaml --resume

# CPU figure regeneration using output_dir and colour limits from the same file.
python scripts/visualize_transcg.py --config configs/transcg_zero_shot.yaml
```

Precedence is **explicit CLI > YAML > built-in defaults**. For example:

```bash
python scripts/run_transcg_zero_shot.py --config configs/transcg_zero_shot.yaml \
    --num-samples 40 --seed 123 --output-dir results/transcg_40_seed123 --no-resume
```

Changing sampling/inference/calibration requires a fresh output directory.
`--no-resume` explicitly overrides YAML `resume: true`. YAML uses flat
underscore keys (e.g. `process_res`), native booleans (`true`/`false`) and
`intrinsics_shape: [HEIGHT, WIDTH]`. Keep both calibration fields `null` until
verified. Unknown keys, malformed YAML and invalid types/values fail before
inference. Relative **paths in YAML** are resolved against `ML_project/`, so
moving the config or changing your working directory does not change their
meaning. Explicit CLI paths retain their normal working-directory semantics.
`--config` is opt-in; existing commands without it retain their defaults. YAML
reading uses PyYAML from the existing OmegaConf/DA3 dependency stack; no packages
are installed automatically. Inspect/render commands read only their applicable
settings from the shared file. Booster `configs/zero_shot.yaml` is unchanged.

Explicit CLI usage is also supported. Replace `/path/to/TransCG` below with the
real dataset location:

```bash
conda activate vggtomast3r
cd ~/Documents/Projects/ML_pro/ML_project
export HF_HUB_OFFLINE=1

python scripts/inspect_transcg.py \
    --data-root /path/to/TransCG

python scripts/run_transcg_zero_shot.py \
    --data-root /path/to/TransCG \
    --checkpoint checkpoints/DA3METRIC-LARGE \
    --scene-start 1 \
    --scene-end 10 \
    --camera 1 \
    --num-samples 20 \
    --seed 42 \
    --process-res 560 \
    --device cuda:0 \
    --precision bf16 \
    --output-dir results/transcg_zero_shot
```

`bash scripts/run_transcg_zero_shot.sh <same arguments>` is an alternative entry
point. Neither command activates or changes an environment. Both HF and
Transformers offline mode are forced by the Python runner before model imports.
A missing local checkpoint is an error, with no remote fallback.

Discovery accepts `<root>/scene1`, `<root>/data/scene1`,
`<root>/transcg/data/scene1`, and `<root>/transcg/scene1`. Ambiguous roots are
rejected. This development machine's inspected layout was
`E:\transcg-data-1\transcg\scene1` through `scene10`; no local dataset path is
hardcoded. `inspect_transcg.py` is read-only, scans/decode-checks RGB images and
reports per-scene counts, rejected files, and example depth/mask types and ranges.
It does not run inference or save a selection.

The default candidate pool contains **every readable `rgb1.png`** recursively
under the requested scenes, ordered by numeric scene then lexicographic path.
Depth/mask files and `rgb2.png` cannot enter this pool. A local
`random.Random(42).sample(pool, 20)` selects 20 distinct images globally without
replacement. Before model loading, the runner saves their ordered absolute and
relative paths, scene/frame/camera IDs, RGB sizes and SHA256 hashes in
`sample_manifest.json`. It prints the counts, seed, and all selected paths.
Unreadable candidates are recorded in the scan report. Candidate counts are
readability counts, not official train/test valid-perspective counts; scene
metadata is not used to silently filter the pool.

Add `--sampling per_scene` to select 2 images from each of the 10 scenes. Other
sample counts must divide evenly over all requested scenes; insufficient images
cause an error. `--camera 2` explicitly selects L515 only and requires a separate
output directory. Cameras are never mixed within a run.

The model loads once, then receives one RGB array per call, using the existing
official DA3 preprocessing and `torch.inference_mode()`. `upper_bound_resize`
preserves the full field of view with patch rounding, without cropping. Times
include official preprocessing, forward and postprocessing, synchronized on the
GPU; CUDA peak allocated memory includes the resident model. They exclude file
saving, figures and evaluation, and include first-call warm-up overhead.

### Depth scale, GT and optional evaluation

`canonical_depth.npy` always stores the untouched float32 native network output;
it is **not labelled metres**. Without confirmed camera intrinsics and their
pixel grid, inference remains available but the program warns that scale is
uncalibrated and trustworthy Raw Metric Evaluation is disabled. It creates no
`metric_depth.npy` or `metrics_summary.json` in that mode.

If genuine `transcg-info` camera intrinsics are available, explicitly confirm
their camera correspondence and original pixel grid. For example, **only after
verifying that D435 K belongs to the original 720x1280 RGB grid**, append:

```bash
    --intrinsics /path/to/transcg/camera_intrinsics/1-camIntrinsics-D435.npy \
    --intrinsics-shape 720 1280
```

Use a fresh output directory when enabling calibration. Alternatively, if the
official `camera_intrinsics/1-camIntrinsics-D435.npy` (or camera 2 equivalent) is
inside the resolved dataset root, `--intrinsics-shape` alone enables discovery.
The shape has no guessed default. Missing info files, or an unconfirmed grid,
leave output canonical; an explicitly supplied missing/invalid K is an error.
An image that differs from the confirmed grid fails metric conversion. K is
scaled by the **actual native output width/height**, including patch rounding:

```text
K_native = diag(W_native/W_rgb, H_native/H_rgb, 1) @ K_rgb
focal_native = (fx_native + fy_native) / 2
depth_m = canonical_depth * focal_native / 300
```

No GT statistics, sensor depths or predicted cameras determine this conversion.
Calibration source, SHA256, K matrices, dimensions, factor and units are saved.

The [official TransCG layout](https://graspnet.net/transcg) distinguishes
`depthN.png` (raw sensor depth), `depthN-gt.png` (refined GT), and
`depthN-gt-mask.png` (transparent objects, nonzero foreground). The
[official dataset reader](https://github.com/Galaxies99/TransCG/blob/main/datasets/transcg.py)
and [depth conversion](https://github.com/Galaxies99/TransCG/blob/main/utils/data_preparation.py)
use **D435 / 1000** and **L515 / 4000** to convert stored depth to metres.
This reader checks 16-bit GT PNG encoding, spatial correspondence, and binary
0/1 or 0/255 masks. Missing/unrecognized annotations produce explicit warnings;
raw sensor depth is never substituted. No GT filling, training normalization,
range/outlier filtering, or mask resizing is applied.

With calibrated predictions and readable GT, `metrics_summary.json` reports raw
AbsRel, RMSE and delta1, globally weighted by pixels, with full-image and
transparent-object regions. Evaluation uses finite positive GT/prediction
overlap, reports coverage and skipped annotations, and performs no GT scale fit.
No metrics file is created when no valid overlap exists. This diagnostic samples
scene1-10 irrespective of official split and does not claim official test scores.

### Outputs, visualization and resume

```text
results/transcg_zero_shot/
  sample_manifest.json
  run_metadata.json
  inference_summary.json
  inference_failures.json
  predictions/sceneN/<stable hashed sample ID>/
    canonical_depth.npy          # native network grid, uncalibrated
    metric_depth.npy             # only when calibration is confirmed
    prediction.json              # input provenance, units, timing, memory, hashes
    predicted_depth.png          # RGB-size prediction colour image
    visualization.json           # colour limits, units, annotation warnings
    failure.json                 # failed attempts; removed on successful retry
  figures/sample_001.png          # one comparison panel per successful sample
  contact_sheet.png               # ordered slots for ALL selected samples
  visualization_summary.json      # panel settings and explicit failures
  metrics_summary.json            # only valid calibrated prediction/GT overlap
```

Without GT the panel is RGB | prediction, plus a transparent mask overlay if
available. With readable GT it is RGB | GT | prediction | absolute error, plus
the optional mask. Without calibrated predictions the error panel explicitly
says unavailable, while canonical colours use the prediction's P2/P98 range
and are labelled **UNCALIBRATED, not metres**. GT still uses metres in its own
labelled range. Metric GT/prediction share viridis and the same fixed range
(default 0-2 m); errors use magma (default 0-1 m), only on valid overlap.
Grey denotes invalid pixels. Saturation affects display only; arrays are intact.
All panels preserve the RGB aspect ratio. Contact-sheet thumbnails preserve the
whole comparison and include numbered failed/unavailable slots, with no
replacement of selected images. Full-size individual panels support edge review.

Review transparent surfaces, background leakage, boundary continuity and scale
deviation from the actual server predictions. The code assumes no conclusion.

Re-run the **same inference command with `--resume`** after interruption. The
saved manifest is reused without rescanning/replacing images. Completed samples
are skipped only after checking RGB hashes, config, array shapes/dtypes and
checksums. Failed samples without a successful marker are retried. A changed
selection/config/checkpoint/calibration requires a new output directory; a
corrupt successful result is reported rather than silently overwritten. Choose
a new output directory to recompute it. Nonempty foreign output directories are
rejected, protecting existing Booster metrics. Figures can be regenerated on CPU:

```bash
python scripts/visualize_transcg.py \
    --output-dir results/transcg_zero_shot --depth-max 2 --error-max 1
```

Synthetic tests and syntax checks do not load DA3 or run GPU inference:

```bash
python -m pytest tests/test_transcg.py tests/test_inference_adapter.py \
    tests/test_metric_scale.py -q --basetemp .test-tmp-transcg
python -m compileall -q src scripts tests
```

Actual checkpoint loading at the pinned DA3 commit, BF16 on RTX 4090D, native
output dimensions/focal conversion, GPU memory/time, server path permissions,
real annotation alignment and resulting transparent-object quality still need
verification on the server. Development did not execute real TransCG inference.
