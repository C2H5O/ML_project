# Executed validation (2026-10-25)

This record distinguishes actual Windows execution from target Linux validation.
No training, fine-tuning or full-data preprocessing was performed. Dataset files
were read only. No depth GT was regenerated.

| Check | Observed result |
|---|---|
| Source/static checks | Python source compiled; Git whitespace check passed; both Bash files passed `bash -n` |
| CPU/unit/integration tests | 21 tests passed under Python 3.10.20 / torch 2.3.1 |
| Processed dataset audit | All 76 indexed samples readable; 0 errors; no missing GT or masks |
| Preprocessing spot checks | Four raw/processed RGB and material pairs matched exactly after the recorded transforms |
| Official API import | Passed at pinned source commit, with exact lazy-export patch |
| Real GPU smoke test | DA3-BASE, 3 images, BF16, 448x560, RTX 4060 Laptop 8GB; passed |
| Full local inference | All 76 train observations succeeded; no failures |
| Full local evaluation | All 76 observations evaluated; 18,667,168 full-image valid pixels |
| Local checkpoint / real resume | HF directory loaded offline, one prediction saved, subsequent run verified and skipped it |
| Report figures | Six actual 3200x1800 PNG panels generated; representative panels visually inspected |
| Target 4090D/A800 | Not accessed or tested |
| Linux / GLIBC 2.17 | Not executed; server setup remains to be validated there |
| DA3-SMALL / DA3METRIC-LARGE | Supported interfaces, no checkpoint download or inference validation |

`pip check` also reported no broken requirements in the validated local venv.

The test suite covers sample correspondence, invalid-depth filtering, material
encoding, missing/unconfirmed data, exact small-array metrics, strict delta1,
scale alignment, abnormal scale/empty masks, global aggregation, array/CSV/JSON
formats, RGB-only API inputs, non-silent failure records, resume checksum checks,
prediction resizing and output protection. The end-to-end unit pipeline uses an
explicit fake predictor; the real checkpoint checks are separately listed above.

## Verified local runtime

- Python 3.10.20, PyTorch 2.3.1 (Conda build), torch CUDA runtime 12.1.
- torchvision 0.18.1, NumPy 1.26.4, Pillow 10.4.0, OpenCV 4.10.0.84.
- matplotlib 3.9.2, einops 0.8.0, huggingface-hub 0.26.2, safetensors 0.4.5,
  OmegaConf 2.3.0, addict 2.4.0, imageio 2.36.0, tqdm 4.67.1, evo 1.29.0,
  scipy 1.13.1. Optional xformers absent and not needed by this model path.
- NVIDIA GeForce RTX 4060 Laptop GPU, CUDA operation and BF16 support checked.
- Windows host; no GLIBC or GCC version can be validated on this host.

See [environment_validation.json](environment_validation.json) for actual import
results. A workspace-local venv reused the existing Python/torch installation
via system site-packages and installed extra pinned runtime packages locally;
the existing shared environment was not upgraded. The server recipe instead
starts from an isolated Conda environment and explicitly installs cu121 wheels.

## Model and run

- Model: `depth-anything/DA3-BASE` (official `da3-base` architecture).
- Weight Hub revision: `f4a6c9b3c95e41c82048423d3493a81ec3fa810e`.
- Official code commit: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
- Process resolution 560; `upper_bound_resize`; official output 448x560.
- CUDA BF16; one image per official API call; seed 42; no GT model input.
- Total synchronized inference sections: about 9.98 seconds for 76 images;
  peak allocated GPU memory about 819.04 MiB. This includes official image
  processing/forward/output processing, and excludes model download/load, disk
  writes, evaluation and plotting. It is a local functional run, not a controlled
  performance benchmark or a prediction of 4090D/A800 throughput.

## Actual scale-aligned results

**Evaluation-time GT alignment; not raw metric depth; evaluated on supplied train
split.** Each image uses a single scale estimated on full valid GT/prediction
overlap. All regions share that scale. Default `validity=all` excludes the separate
stereo consistency mask. Metrics accumulate over global valid pixels.

| Region | Images with pixels | Valid pixels | AbsRel | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|---:|
| Full image | 76 | 18,667,168 | 0.0559773 | 0.1087464 | 0.9622909 |
| Target classes 2+3 | 64 | 2,831,089 | 0.0625415 | 0.1565896 | 0.9579639 |
| Ordinary class 0 | 76 | 10,273,658 | 0.0562988 | 0.1074334 | 0.9499563 |
| Other classes 0+1 | 76 | 15,836,079 | 0.0548038 | 0.0977564 | 0.9630644 |
| Material 1 | 70 | 5,562,421 | 0.0520426 | 0.0767393 | 0.9872748 |
| Material 2 | 24 | 1,193,693 | 0.0434002 | 0.0556772 | 0.9992192 |
| Material 3 | 48 | 1,637,396 | 0.0764959 | 0.2003400 | 0.9278879 |

Machine-readable numeric evidence (including the class-0 duplicate row, full
precision, per-image scales/coverage and empty regions) is committed under
[zero_shot_results](zero_shot_results/metrics_summary.json). Raw predictions and
figures are local, under `results/zero_shot`, and deliberately ignored by Git.
No image/checkpoint/cache or large dataset is part of the commit.

## Issues found and practical limits

1. The existing writer saved scalar masks as three identical channels; the
   loader now verifies that encoding rather than assuming grayscale PNGs.
2. Official nonmetric output can carry an empty AddictDict as `is_metric` due
   to upstream `getattr` behaviour. It is stored as unknown (`null`), while the
   verified requested model architecture defines relative-depth semantics.
   No prediction arithmetic is changed to fix metadata.
3. Local Conda CUDA NVRTC builtins required its existing `bin` directory in the
   process PATH. Only that process's environment was changed; no DLL was copied,
   replaced or modified. An initial failed smoke run was rerun successfully.
4. Windows PyTorch warned that flash attention was unavailable; ordinary PyTorch
   SDPA executed successfully. No attention extension was compiled or installed.
5. Preprocessing retains scene-dependent `mtxL` versus `proj_matL` focal
   discrepancies (49 calibration warnings). The metric values are conditional
   on this supplied GT/calibration convention. No claim of independent physical
   depth calibration is made.
6. Raw train/test directory names overlap at `Cooker1`. This is a potential
   leakage warning, not proof of an identical physical scene; resolve it before
   constructing future fine-tuning partitions. Processed data currently contains
   only train, so no cross-split processed evaluation was performed. This adds
   one audit warning to the 49 calibration warnings.
7. Full local evaluation is not target-server execution, held-out testing,
   separate transparent/mirror semantic evaluation, or an aligned-free metric
   baseline. No such claim is made.

To reproduce on the server use the README commands and add
`--revision f4a6c9b3c95e41c82048423d3493a81ec3fa810e` to pin the tested weights.
