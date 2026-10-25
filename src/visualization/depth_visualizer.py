"""Report panels; colour limits are explicit and raw arrays never mutated."""
from pathlib import Path
import numpy as np


def save_panel(sample, raw, path, vmin=0.0, vmax=5.0, error_max=1.0, dpi=200,
               validity="all"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from src.evaluation.metrics import valid_pixels
    gt, valid = sample["depth_gt"], sample["valid_mask"].copy()
    if validity == "consistent":
        valid &= sample["consistency_mask"]
    comparison_valid = valid_pixels(raw, gt, valid)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(sample["rgb"])
    axes[0].set_title("RGB input (saved processed image)")
    depth_panels = [(1, np.where(valid, gt, np.nan), "Ground truth (m)"),
                    (2, np.where(np.isfinite(raw) & (raw > 0), raw, np.nan), "Raw metric DA3METRIC-LARGE (m)")]
    if comparison_valid.any():
        error = np.full(gt.shape, np.nan)
        relative = np.full(gt.shape, np.nan)
        error[comparison_valid] = np.abs(raw[comparison_valid] - gt[comparison_valid])
        relative[comparison_valid] = error[comparison_valid] / gt[comparison_valid]
        im = axes[3].imshow(error, cmap="magma", vmin=0, vmax=error_max)
        fig.colorbar(im, ax=axes[3], label="Absolute raw metric error (m); saturated at vmax")
        axes[3].set_title("Absolute error (no GT alignment)")
        im = axes[4].imshow(relative, cmap="magma", vmin=0, vmax=1)
        fig.colorbar(im, ax=axes[4], label="Absolute relative error; saturated at 1")
        axes[4].set_title("Relative error (no GT alignment)")
    else:
        axes[3].text(.1, .5, "Absolute error unavailable", transform=axes[3].transAxes)
        axes[4].text(.1, .5, "Error unavailable: no valid GT/prediction overlap", transform=axes[4].transAxes)
    for index, depth, title in depth_panels:
        im = axes[index].imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[index].set_title(title)
        fig.colorbar(im, ax=axes[index], label="Depth (m); shared numeric range")
    material = sample["material_mask"]
    axes[5].imshow(sample["rgb"])
    axes[5].set_title("Material severity overlay")
    if material is None:
        axes[5].text(.1, .5, "Material mask unavailable", transform=axes[5].transAxes)
    else:
        colours = np.array([[.65,.65,.65], [.95,.75,.1], [.95,.35,.1], [.6,.15,.9]])
        axes[5].imshow(colours[material], alpha=.45)
        axes[5].legend(handles=[Patch(color=colours[i], label=f"class {i}") for i in range(4)], fontsize=8)
    for axis in axes:
        axis.axis("off")
    fig.suptitle(sample["sample_id"] + " | Raw metric evaluation: calibrated focal/300, no GT alignment", fontsize=14)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
