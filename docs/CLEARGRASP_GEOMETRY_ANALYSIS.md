# Saved ClearGrasp geometry analysis

This workflow uses the existing 20 real-test predictions from DA3METRIC-LARGE.
It never loads a model, changes saved prediction values, samples new images or
writes the original `metrics/` directory. Development validation uses synthetic
small fixtures only. Real scores and conclusions require running on the server.

## Paths and migration

Manifest identity `d415/000000029` with split `real-test`, or the equivalent
full ID `real-test/d415/000000029`, maps to:

```text
<output_dir>/predictions/real-test/d415/000000029/
  canonical_depth.npy
  metric_depth.npy
  prediction.json
<output_dir>/figures/real-test/d415/000000029/
  comparison.png
  ... existing image files ...
```

Old directories are resolved with the existing SHA-256(sample ID)[:24] naming
algorithm. Both relative-ID and split-prefixed-ID hashes are supported, but
multiple existing candidates are an error: the loader never silently chooses
between conflicting copies. IDs validate split, camera/group and numeric sample
number, reject traversal/absolute paths and Windows reserved names, and reject
case-insensitive duplicate identities in the manifest. Existing links/reparse
points inside result paths are rejected. The shared `sample_output()` is unchanged.

Run the exact commands in the README in your existing `vggtomast3r` environment.
No migration flag means dry-run. `--dry-run` validates all 20 samples, hashes and
metric provenance before printing source/destination mappings; it creates no
files. `--apply` uses the same complete preflight, then moves whole prediction
and figure directories. It refuses existing destinations, verifies all file
checksums again, and records the mapping and progress in `path_migration.json`.
NPY and JSON bytes are never rewritten. Missing legacy figures are recorded as
absent, not fabricated. Repeating a completed migration returns `already_readable`.

Stop concurrent inference, plotting, migration and analysis before moving paths.
On an exception the tool attempts reverse-order rollback and records failures.
For a crash or `rollback_incomplete`, preserve the journal and both directory
trees, inspect the reported conflict, then use:

```bash
python scripts/migrate_cleargrasp_result_paths.py \
    --config configs/cleargrasp_zero_shot.yaml \
    --recover
```

Recovery validates manifest-derived paths and planned hashes. It refuses to
guess if both source/destination exist or both are missing. An abrupt process
kill may leave `.cleargrasp_migration.lock`; only after confirming no writer is
running, remove that lock file and retry recovery. Do not delete arrays or the
journal to resolve a conflict. Migration should precede any optional redraw of
old comparison figures, because a redraw creates the new readable figure path.

Old contact-sheet and visualization-summary files are intentionally preserved
byte for byte; they can contain historical hash paths. The migration journal
maps those paths, and the new analysis report uses readable links throughout.

## Configuration and execution

Retain the current server YAML and output directory containing the saved
manifest. Checkpoint/device settings can stay unchanged: the analyzer ignores
model loading and CUDA. No environment-variable paths are supported. Optional
analysis settings may be added to that same YAML:

```yaml
analysis_expected_samples: 20
analysis_min_pairs: 50          # distinct unordered pairs, not repeated draws
analysis_min_unique_pixels: 10 # required independently at each endpoint
analysis_boundary_width: 4
analysis_min_region_pixels: 10
analysis_baseline_rtol: 0.000001
analysis_baseline_atol: 0.00000001
```

Existing `seed`, `pair_draws`, `gt_gap_m`, `ring_radius`, `boundary_margin`,
`compression_ratio`, `depth_max`, `error_max` and `split` apply. Boundary settings
must satisfy `0 <= boundary_margin < analysis_boundary_width <= ring_radius`.
Defaults are 20,000 candidate draws, 0.02 m strict GT gap, radius 30 px,
margin 2 px and compression ratio 0.25. These are diagnostic protocol settings,
not parameters tuned on the actual test set. `max_samples`/`all` do not select
analysis samples: the saved manifest defines the entire analysis set.

The analyzer validates all selected arrays before computing scores. It checks
sample identity, recorded output hashes, positive finite depth support, metre
units, exactly-once focal conversion provenance, RGB hashes/dimensions and
current calibration against recorded source/hash/K. It records GT and mask hashes.
Any failed sample produces a validation-failure report with zero analyzed
images; there is no silently reduced aggregate. Existing successful or failed
`geometry_analysis/` reports are not overwritten. To deliberately rerun, archive
or rename only that report directory, keeping the saved manifest/predictions
and original metrics in place. Unexpected processing failures preserve a
`.geometry_analysis-<run-id>.partial` directory with error details, not a
misleading completed report.

## Metrics and interpretation

The implementation reuses `MetricAccumulator`, `oracle_scale`,
`region_diagnostics`, `relative_geometry`, `local_pairs`, `ordinal_result`,
saved-prediction loading and the existing resize-to-GT operation.

| Analysis | Definition and interpretation |
|---|---|
| Raw Metric | Full/T/B AbsRel, RMSE and δ1, pooled by valid pixel count across images. Only finite positive GT AND prediction. No GT scaling. |
| Global scale | Compute median(pred/GT) and median(GT/pred) independently per image. The latter is one multiplicative oracle factor, not a shift or per-region fit. |
| Log residual | Natural-log median bias; SILog = sqrt(mean((log(pred/GT) − mean(log(pred/GT)))²)), without ×100; MAD around median log residual. |
| Background alignment | One scale = median(GT/pred) on valid nontransparent pixels only, applied to the whole image. Report T/B residual AbsRel, T−B difference and (T−B)/B; zero B makes relative increase unavailable. |
| Ordinal | Full, T/T, local T/B and B/B sign agreement, excluding identical endpoints and GT gaps ≤ threshold. No GT scaling. |
| Local penetration diagnostic | GT-front T/B subset only: reversal if log(pred B/T) < 0; compression if 0 ≤ predicted log gap < ratio × GT log gap. The disjoint union is combined error. |
| Boundary | Distance-transform interior, safe inner boundary band and nearby background ring. Local normalized contrast 2(B−T)/(B+T), absolute log-contrast error and order reversal. |

The full-image oracle and background alignment are always
**GT-Dependent Diagnostic Only**. Per-image diagnostic factors precede pixel
pooling; reduction in AbsRel is descriptive, not a causal decomposition or a
deployable model score. Median ratios are not assumed to be exact reciprocals.

T/B candidate offsets lie within the configured Euclidean radius and outside
the mask's boundary margin, without a fallback to distant background. Stable
sample-specific seeds make reruns reproducible. Sampling uses replacement;
CSV rows include candidate/accepted draws, unique pairs, unique endpoint counts
and endpoint coverage. Minimum support applies per image before pooling rates.
Unsupported rates are `null`/`unavailable`, not zero or 100%. Many correlated or
repeated pairs do not establish independent statistical significance.

Interior distance exceeds the boundary width; the safe boundary band lies
between the exclusion margin and width. Image-edge bands are excluded from
boundary pairing. All three regions need enough valid pixels for contrast;
empty/all-object/thin masks without reliable interior or background produce
unavailable contrasts. Normalized and log contrasts are invariant to positive
global scale, as are ordinal/reversal/compression metrics. Nearby background
is not the occluded background on the same ray: this diagnostic cannot prove
or disprove physical surface penetration.

## Output and comparison with the original baseline

```text
<output_dir>/geometry_analysis/
  summary.json
  validation.json
  baseline_comparison.json
  raw_metric_recomputed.json
  per_image.json
  per_image.csv
  scale_analysis.csv
  ordinal_analysis.csv
  penetration_analysis.csv
  boundary_analysis.csv
  report.md
  figures/
    scale_ratio_distribution.png
    raw_vs_aligned_absrel.png
    transparent_vs_background.png
    ordinal_accuracy.png
    penetration_rates.png
    hardest_samples.png
    real-test/d415/000000029/analysis.png
    ... every saved sample ...
```

`summary.json` includes original manifest/baseline hashes, counts, protocol,
pixel-weighted raw/diagnostic metrics, per-image scale distributions,
support-qualified pair summaries and hardest full sample IDs. CSVs contain
flattened named fields; missing numeric fields are blank, with explicit status
and support columns. `report.md` derives its observations from these computed
values without assuming the model is good or bad. Hardest lists include up to
three samples for each requested criterion, excluding unavailable metrics.

Every sample gets RGB, GT, full unmasked raw prediction, background-aligned
depth, raw error, background-aligned error, mask, local ordinal error and
boundary log-contrast error. GT/raw share fixed metre colour limits. Display
saturation never clips NPY data. Diagnostic panels carry the explicit label.
Pair heatmaps average errors at sampled transparent endpoints; grey includes
unsampled positions and unavailable comparisons, not just invalid depth.

The analyzer compares recomputed counts and metrics with the existing
`metrics/raw_metric_summary.json`, reports differences and never overwrites it.
Tolerances are recorded. Exit 0 means analysis completed (baseline may be absent,
explicitly marked unavailable); exit 2 means failed sample validation or
baseline mismatch. Other processing failures raise an error and preserve the
partial report. Investigate mismatches before citing results. No server-derived
scale, ordinal or penetration numbers are supplied by the development tests.

## Synthetic regression checks

```bash
python -m pytest -q --basetemp .test-tmp-cleargrasp-geometry
```

Tests cover safe/readable paths, legacy loading, migration byte preservation,
conflicts/checksum failure, rollback/recovery, idempotence, all-20 validation,
CPU-only imports, report artifacts and baseline preservation, raw versus scale
diagnostics, background-only scale estimation, median-ratio definitions,
positive-scale invariance, reversals and unreliable boundary masks. The full
suite also runs the existing Booster/TransCG regression tests.
