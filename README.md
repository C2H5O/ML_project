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

Historical BASE predictions/results remain distinguishable at their original
locations and commit. They are not results for this Metric-Large protocol. See
[the detailed protocol](docs/ZERO_SHOT_GUIDE.md) for audit limitations and tests.
