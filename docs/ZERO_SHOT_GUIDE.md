# Protocol and reproducibility notes

Use the executable Linux commands in [README](../README.md). CLI `--help` is
available on all Python entry points. `configs/zero_shot.yaml` documents defaults;
it is not implicitly loaded and cannot silently override CLI choices.

## Model/API provenance

Official repository: https://github.com/ByteDance-Seed/Depth-Anything-3

Inspected/pinned source commit: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.

Official API: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/docs/API.md

The wrapper calls `DepthAnything3.from_pretrained`, `eval`, `inference_mode`, and
`model.inference(image=[uint8_rgb], process_res=..., process_res_method=...,
infer_gs=False, export_dir=None)`. `Prediction.depth` is read as `[1,H,W]`.
Normalisation happens inside the official InputProcessor exactly once. Only
single-view calls are made. Ground truth and intrinsics are never passed to DA3.

Two explicit compatibility accommodations are used:

1. `patch_da3_exports.py` moves exporter imports into their existing branches,
   avoiding mandatory open3d/pycolmap/moviepy/GS imports during depth-only use.
   The script checks the exact source commit, rejects unknown edits, and can
   verify its own idempotent patch. No model/preprocessing code is edited.
2. The model instance's `forward` AMP wrapper preserves the exact underlying
   official network call but obeys requested precision. Upstream hardcodes an
   autocast policy, so an outer user autocast alone is insufficient. Model
   parameters stay float32; CUDA AMP is explicit, CPU uses float32. All numerical
   precision choices are stored in run metadata.

The dependency set covers imports required by API, preprocessing, dual DPT,
DINOv2, pose alignment, and output processing, plus evaluation/plots/tests.
Optional exporters are outside this baseline. Installation is constrained so
PyTorch/torchvision cannot be upgraded by pip resolution. The resolver may choose
compatible transitive packages; save `pip freeze` for each server environment.
No GLIBC or CUDA system modification, xformers compilation or GS installation
is required. A Windows smoke test cannot certify CentOS/GLIBC binary compatibility.

## Data/metric caveats

The supplied metric GT comes from a previous preprocessing job and is authoritative
for this stage. Its calibration choice is recorded, with focal disagreements
retained as warnings. Recalibrating or regenerating depth would define a new GT
version and is outside this baseline. Physical GT correctness is not inferred
solely from finite values or successful shape checks.

Material semantics follow the [Booster paper](https://arxiv.org/html/2301.08245v3),
Section III: IDs 0 through 3 are increasing specularity/transparency levels.
Class 0 is opaque/diffuse; class 1 has heavily distorted reflected/transmitted
appearance; class 2 retains partly distorted recognizable appearance; class 3
is regular reflection/transmission. There are no independently confirmed
transparent vs mirror semantic labels in the processed 0–3 mask. Such separate
scores are unavailable. Target=2+3 is an explicit project protocol, and all
four severity classes are included so no grouping information is hidden.

Only `train` is currently processed. Zero-shot evaluation on it is permissible
because this stage does not train, but those values are not a held-out test
benchmark. Later head/LoRA experiments must first freeze a scene-level partition,
group paired balanced/unbalanced observations and avoid duplicate input leakage.

Default aggregation is global pixel-weighted across selected observations. The
two stereo setups are separate GT observations of the same scene and must not
be mistaken for twice as many independent scenes. `--subset balanced` or
`--subset unbalanced` is available to report either setup separately. Missing
GT/masks or unknown units/IDs produce explicit failures rather than guessed scores.

## Reproducible comparison

Keep model Hub revision, official code commit, data/audit version, split, sample
selection, process resolution/method, precision, validity protocol and region
definitions fixed across baselines. Check `metrics_summary.json` completeness,
per-region sample/pixel counts and per-image prediction coverage before using
numbers in a report. Seed is recorded; deterministic bitwise GPU equivalence
across different drivers/devices is not guaranteed.

The figures use the same fixed numeric limits for GT, raw and aligned depth,
while explicitly distinguishing raw relative units from metres. Their error
panel uses GT-aligned prediction only. For a future metric model, raw metric
numeric evaluation is supported; the current figure layout remains explicitly
GT-aligned and should not be used as evidence of metric calibration.
