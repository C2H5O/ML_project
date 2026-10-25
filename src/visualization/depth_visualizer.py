"""Report panels; colour limits are explicit and raw arrays never mutated."""
from pathlib import Path
import numpy as np


def save_panel(sample, raw, path, vmin=0.0, vmax=5.0, error_max=1.0, dpi=200,
               validity="all"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from src.evaluation.metrics import align_scale, valid_pixels
    gt, valid = sample["depth_gt"], sample["valid_mask"].copy()
    if validity == "consistent":
        valid &= sample["consistency_mask"]
    try:
        aligned, scale = align_scale(raw, gt, valid)
        comparison_valid = valid_pixels(aligned, gt, valid)
    except ValueError:
        aligned, scale = None, None
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(sample["rgb"])
    axes[0].set_title("RGB input (saved processed image)")
    depth_panels = [(1, np.where(valid, gt, np.nan), "Ground truth (m)"),
                    (2, np.where(np.isfinite(raw) & (raw > 0), raw, np.nan), "Raw DA3 (model-relative units)")]
    if aligned is not None:
        depth_panels.append((3, np.where(comparison_valid, aligned, np.nan),
                             f"GT scale-aligned (m), scale={scale:.4g}"))
        error = np.where(comparison_valid, np.abs(aligned - gt), np.nan)
        im = axes[4].imshow(error, cmap="magma", vmin=0, vmax=error_max)
        fig.colorbar(im, ax=axes[4], label="Absolute aligned error (m); saturated at vmax")
        axes[4].set_title("Absolute error after evaluation-time GT alignment")
    else:
        axes[3].text(.1, .5, "GT alignment unavailable", transform=axes[3].transAxes)
        axes[4].text(.1, .5, "Error unavailable: no valid GT/prediction overlap", transform=axes[4].transAxes)
    for index, depth, title in depth_panels:
        im = axes[index].imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[index].set_title(title)
        fig.colorbar(im, ax=axes[index], label="Depth (raw uses relative units); shared numeric range")
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
    fig.suptitle(sample["sample_id"] + " | GT-aligned results are not raw metric predictions", fontsize=14)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
