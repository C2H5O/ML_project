# Observed processed Booster structure

```text
<root>/train/index.json
<root>/train/{balanced,unbalanced}/<scene>/camera_00/
  rgb.png, depth_gt.npy, valid_mask.png, mask.png, material_mask.png, meta.json
<root>/_pointcloud/, _pred_gt/, train/_viz/ (derived outputs; ignored)
```

This report is from actual file reads. The data root is intentionally omitted.

GT is saved metric depth in metres, not disparity. Loader performs no resize or unit conversion.

Each indexed subset/scene/camera ID pairs all six files through meta.json/files.

File counts by role/extension: {'rgb:.png': 76, 'depth_gt:.npy': 76, 'mask:.png': 76, 'valid_mask:.png': 76, 'meta:.json': 76, 'material_mask:.png': 76}
Missing files: 0

- Split `train`: 76 indexed; 76 readable.
- Subsets: {'balanced': 38, 'unbalanced': 38}
- Shapes/dtypes: ["([448, 560], 'float32')"]
- Material pixel counts: {1: 5717173, 2: 1201142, 3: 1660969, 0: 10487596}
- Depth range (m): 0.410430282 to 4.241902828

Errors: 0; warnings: 50 (details in dataset_audit.json).

Valid pixels: saved valid_mask AND finite GT AND GT > 0

Separate stereo consistency mask; excluded from default validity

Material labels describe severity, not separate transparent and mirror classes:

- 0: opaque_or_diffuse
- 1: strongly_distorted_reflection_or_transmission
- 2: partially_distorted_reflection_or_transmission
- 3: regular_specular_or_transparent

Definition: [Booster paper](https://arxiv.org/html/2301.08245v3#S3).
Project protocol: target = 2+3, ordinary = 0, additional other = 0+1; all four classes are also reported.

Only train is currently indexed. No validation/test split is invented. Balanced/unbalanced views
share scene identities and must stay together in any future train/validation/test partition.

Preprocessing source inspection: original 3008x4112 -> area RGB/depth resize 448x612 -> centre crop
(x=26,y=0) to 448x560. Depth uses valid-aware area averaging; material uses nearest neighbour.
K uses actual scale_x/scale_y and crop offsets. Original disparity was converted once using
fx(mtxL)*baseline_mm/disparity/1000 before resizing. Saved GT is reused unchanged.

Known calibration caveat: mtxL and proj_matL focal lengths differ in the existing preprocessing
(scene-dependent; see exact warnings in the JSON). These measurements are conditional on the existing
mtxL-based GT; this audit does not establish which calibration model is physically preferable.

Mask PNGs are stored as RGB with identical channels by the existing writer; the loader validates
channel equality before reducing to one channel. No class values are inferred from colour.

Only processed indexed splits audited; absence of test GT is not a test evaluation
Metadata K transform verified, not physical camera calibration accuracy
Existing mtxL vs projection focal discrepancy retained; no GT recomputation

Raw split name check (no depth conversion/reprocessing):
{'test:train': ['Cooker1']}
Name overlap is a leakage warning, not proof of identical physical scenes/images
Do not reuse overlapping scene names in separate fine-tuning splits without resolving this warning.
