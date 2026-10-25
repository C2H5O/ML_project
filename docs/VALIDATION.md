# Metric-Large validation status

This is the current DA3METRIC-LARGE protocol. Previous DA3-BASE scale-aligned
results under docs/zero_shot_results and docs/environment_validation.json are
historical records from commit 866648a; they are NOT Metric-Large results.

The server environment name is vggtomast3r. The server forbids SSH, so this
session cannot connect to it or run RTX 4090D inference. No server metrics,
checkpoint loading success or completed Metric-Large GPU experiment is claimed.
No new environment was created and no dependency was installed/upgraded in this
change. Existing local .venv was reused for CPU and source-only tests.

Executed checks and their exact results are recorded in metric_large_validation.json
(34 tests passed; 14 Matplotlib/pyparsing deprecation warnings, no test failures):

- Unit and pipeline tests using synthetic arrays, including GT-independence.
- Actual official InputProcessor and apply_metric_scaling, without weights.
- Synthetic K/output-grid scale checks against actual official code.
- After the G drive was remounted, all 76 real samples passed dataset re-audit
  and official calibration/scaling comparison (no weights, no GT fit).
- Python compilation and Git whitespace checks.
- Shell syntax checks of the read-only preflight and existing-env runner.

Server-only pending checks:

1. Existing vggtomast3r Python/PyTorch/torchvision/CUDA/GLIBC/source imports.
2. Selected RTX 4090D, BF16 capability and real checkpoint compatibility.
3. Three-image Metric-Large smoke inference and raw metric evaluation.
4. Full selected split, per-material metrics and report figures.

Do not reuse earlier BASE numbers as Metric-Large results. No model weights are
downloaded or model inference is performed on another GPU as a substitute for
the requested 4090D execution. The final user-confirmed server runner defaults
to process_res=560, preserving 448x560 input, and verifies GPU name before loading
the checkpoint; figures contain no GT alignment.
