# Adapting Depth Anything 3 to Transparent and Reflective Indoor Scenes

Reproducible **single-image, zero-shot DA3** baseline on the existing processed
Booster schema. Default model: `depth-anything/DA3-BASE`; switch to
`depth-anything/DA3-SMALL` with `--model-id`. This stage performs inference only.

DA3-BASE/SMALL output relative depth. The default reported protocol is
**evaluation-time per-image GT scale alignment**, not raw metric prediction.
Raw arrays are always retained. No GT, material masks, valid masks or camera
parameters enter the model. No training, multiview fusion or Gaussian Splatting.

## Data inspected

See [the observed structure](docs/DATASET_STRUCTURE.md),
[machine-readable audit](docs/dataset_audit.json), and
[raw/processed spot checks](docs/preprocessing_spot_checks.json).

- Existing `train/index.json`: 76 samples, balanced 38 + unbalanced 38,
  representing 38 scene names, one `camera_00` sample per scene/subset.
- Saved RGB, metric `float32` depth and all masks are 448x560 (H,W).
- All indexed GT and mask files are readable; saved valid masks exactly match
  finite positive GT. Global observed depth range: 0.410430282–4.241902828 m.
- Masks are PNGs with three identical channels from the existing writer. The
  loader validates equality then reads the scalar IDs. They are not RGB palettes.
- Material IDs encode increasing transparency/specularity severity (0–3),
  **not** separate glass/mirror labels. Per-class results are always reported.
  Project grouping: target=2+3, ordinary=0, additional other=0+1.
- Only processed `train` has an index/GT here. Do not claim held-out test scores.
  Never split paired balanced/unbalanced scene identities across training and
  validation/testing in later fine-tuning work.
- Original preprocessing used `mtxL` focal lengths for disparity-to-depth;
  these differ from projection-matrix focal lengths for some scenes. Warnings
  are preserved in the audit. This baseline reuses the supplied GT unchanged.

The loader does **no** depth conversion, normalisation, resize or crop. Optional
`--scene NAME` (repeatable) and `--subset balanced|unbalanced` select existing
samples without creating a new split. A nonexistent split fails explicitly.

## Linux server environment

Create an isolated environment; do not update a shared Python environment:

```bash
git clone --branch zero_shot https://github.com/C2H5O/ML_project.git
cd ML_project
conda env create -f environment.yml
conda activate da3-booster
# If needed on your server, load the provided GCC 9.4.0 module by its actual name.
nvidia-smi
bash setup_env.sh
```

This pins Python 3.10.20, torch 2.3.1+cu121, torchvision 0.18.1+cu121 and
NumPy 1.26.4. CUDA 12.4 server drivers can run a CUDA 12.1 wheel runtime;
the setup never requests a nonexistent torch 2.3.1+cu124 wheel. Check the
actual driver and small CUDA operation using `check_environment.py` first.
No system GLIBC, driver, toolkit or shared library is changed.

`setup_env.sh` installs constrained runtime dependencies, clones official DA3
at `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, applies the small
[lazy-export import patch](scripts/patch_da3_exports.py), and registers its
`src` directory via a `.pth` file. It deliberately avoids upstream package
dependency resolution: **no** unconstrained `pip install -e .`, no optional
gsplat/FlashAttention/xformers build. Actual DA3 depth modules at this revision
use PyTorch scaled-dot-product attention and do not need xformers.

The patch moves optional exporter imports into their existing dispatch branches;
it does not modify a network, weight, preprocessing or output computation.
The API's AMP wrapper is controlled per model instance so `--precision fp32`,
`fp16` and `bf16` actually select the requested precision. CPU uses fp32;
CUDA auto prefers BF16, falling back to FP16 if BF16 is unsupported. Unsupported
explicit precision requests fail rather than silently change precision.

```bash
python scripts/check_environment.py --strict --import-da3 --output logs/environment.json
python -m pytest -q
```

Logs include resolved dependency versions and the DA3 source diff. The Windows
validation described in [VALIDATION.md](docs/VALIDATION.md) does not establish
Linux GLIBC 2.17 compatibility. Run the strict check on the target server;
dependency installation/ABI validation must succeed there before the full run.

## Transfer the existing processed data

The Windows disk is not automatically available on Linux. Copy the **already
processed** Booster directory to a server location with enough space (for example
with rsync, scp, or your lab's transfer client); preserve `train/index.json` and
all paths below `train/{balanced,unbalanced}`. No raw-data download or preprocessing
is needed. The following path is a placeholder you must replace:

```bash
export DATA_ROOT=/absolute/path/on/server/processed/booster
python scripts/inspect_booster.py --data-root "$DATA_ROOT" --output-dir logs/server_audit
```

For Windows the same CLI accepts `--data-root "G:\processed\booster"`.
Keep result directories outside the dataset root. No dataset files are written.

## Run a 3-image smoke test, then inference

```bash
mkdir -p logs
set -o pipefail
python scripts/run_zero_shot.py \
  --data-root "$DATA_ROOT" --split train \
  --model-id depth-anything/DA3-BASE \
  --output-dir results/smoke --process-res 560 --max-samples 3 \
  --device cuda:0 --precision auto --seed 42 --resume \
  2>&1 | tee logs/smoke.log

python scripts/evaluate_zero_shot.py \
  --data-root "$DATA_ROOT" --split train --max-samples 3 --output-dir results/smoke
python scripts/visualize_zero_shot.py \
  --data-root "$DATA_ROOT" --split train --max-samples 3 --output-dir results/smoke

# Evaluate the supplied train split as the zero-shot baseline, not as held-out test.
python scripts/run_zero_shot.py \
  --data-root "$DATA_ROOT" --split train \
  --model-id depth-anything/DA3-BASE --output-dir results/zero_shot \
  --process-res 560 --device cuda:0 --precision auto --seed 42 --resume \
  2>&1 | tee logs/inference.log
python scripts/evaluate_zero_shot.py \
  --data-root "$DATA_ROOT" --split train --output-dir results/zero_shot \
  2>&1 | tee logs/evaluation.log
python scripts/visualize_zero_shot.py \
  --data-root "$DATA_ROOT" --split train --output-dir results/zero_shot --max-samples 5 \
  --depth-min 0 --depth-max 5 --error-max 1 --dpi 200
```

To select report examples instead of the first five images, repeat `--scene`,
for example `--scene Bathroom --scene Mirror --scene Bottle --max-samples 6`.
Keep scene/subset/max-sample selections consistent across inference, evaluation
and visualisation. `run_zero_shot.sh` runs all three stages over the selected
split; use the explicit commands above for subset/scene/smoke selections.

The default 560 upper-bound resize preserves the saved 448x560 field of view
and is patch-compatible (14). `--process-res-method lower_bound_resize` is also
supported. Crop policies are intentionally rejected because blindly stretching
a cropped prediction back to the GT grid would invalidate correspondence.
For OOM, use a new output directory and lower `--process-res` to 392 or 280,
or use DA3-SMALL. No OOM retry silently changes your experiment settings.

The first online run downloads official weights. The model Hub ref is resolved
to a commit SHA and recorded; optionally supply `--revision SHA`. For offline
servers download both files in advance:

```bash
huggingface-cli download depth-anything/DA3-BASE \
  config.json model.safetensors --local-dir checkpoints/DA3-BASE
# Add this to the inference command (model-id must match the config architecture):
# --checkpoint checkpoints/DA3-BASE
```

`--checkpoint` accepts an HF directory containing `config.json` and
`model.safetensors`, not an arbitrary standalone `.pth`. Local file SHA256 hashes
are recorded. DA3METRIC-LARGE is accepted explicitly as a future interface;
it is never selected or downloaded by default. It has not been validated here.

## Evaluation protocol and outputs

All regions share **one scale per image**, computed from the full valid overlap:

```text
valid = saved_valid_mask & isfinite(gt) & (gt > 0) & isfinite(raw) & (raw > 0)
scale = median(gt[valid]) / median(raw[valid])
aligned = raw * scale
```

There is no shift fit, inverse-depth fit, extra normalisation or per-region fit.
Empty overlap, zero/underflow median and abnormal/nonfinite scales fail visibly.
The alignment scale is an evaluation artifact using GT, not model calibration
available at deployment. It must not select future training hyperparameters
using test GT. For an actual metric checkpoint only, `--mode raw_metric` evaluates
without alignment; the evaluator refuses that mode for DA3-BASE/SMALL.

AbsRel and delta1 accumulate their sums/counts over all valid pixels. RMSE is
the square root of the **global** squared error sum divided by the global count,
not the average of per-image RMSE. Delta1 uses the strict `< 1.25` threshold.
Empty regions have JSON `null` metrics and `status=unavailable`; CSV cells stay
empty. Per-image CSV includes GT eligible pixels and prediction coverage, so
invalid predictions cannot disappear without a coverage record.

Default validity uses saved validity plus finite positive GT. The separately
saved stereo consistency `mask.png` is not a material/foreground mask. To add
it to validity use `--validity consistent` for both evaluation and figures;
on unbalanced samples the original preprocessor may have used depth validity as
its fallback consistency mask. Do not mix these protocols in one comparison.

Outputs below `--output-dir`:

```text
run_metadata.json                    # model revision/source commit/precision/seed
inference_summary.json
inference_failures.json
predictions/<split>/<hashed ID>/
  prediction.json                    # sample ID, RGB hash, shapes, timing, VRAM, array hashes
  raw_native.npy                     # float32 at official output resolution
  raw.npy                            # float32 bilinearly mapped to saved RGB/GT size
  scale_aligned.npy                  # evaluation artifact only
  alignment.json                     # scale, protocol, source hashes
metrics_per_image.csv
metrics_summary.json
metrics_summary.csv
failures.json                        # inference and evaluation failures
figures/*.png                        # 6 panels, default 3200x1800
visualization_settings.json
visualization_failures.json
```

GT and masks are never resized at evaluation. Prediction resizing is continuous
bilinear with invalid support excluded; raw native output remains untouched.
Figures use fixed shared numeric depth limits, explicitly label raw relative
units, and show GT-aligned error in metres. Out-of-range values saturate the
colourbar; numeric predictions are unchanged.

`--resume` verifies configuration, RGB and output checksums and shapes before
skipping a sample. Run configurations cannot be mixed in one result directory.
Errors are recorded with nonzero exit status; partial evaluation is marked
`completeness=partial`. Results/checkpoints/data/caches are ignored by Git.

See [the detailed guide](docs/ZERO_SHOT_GUIDE.md) for reproducibility and known
limits, and [validation evidence](docs/VALIDATION.md) for what actually ran.
