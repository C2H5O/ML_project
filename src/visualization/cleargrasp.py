"""Fixed metre colour limits across samples; grey invalid annotations."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from src.common import read_json, write_json, sample_output, protect_dataset
from src.datasets.cleargrasp import annotations, read_rgb, path_for
from src.evaluation.cleargrasp import saved_prediction
from src.evaluation.relative_geometry import oracle_scale
from src.evaluation.metrics import valid_pixels
from src.inference.da3_zero_shot import resize_prediction


def render_sample(output, sample, rgb, gt, mask, pred, metric, depth_max=2., error_max=0.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if depth_max <= 0 or error_max <= 0:
        raise ValueError('Colourbar limits must be positive')
    folder = sample_output(output, sample['split'], sample['sample_id'])
    target = Path(output) / 'figures' / sample['split'] / folder.name
    target.mkdir(parents=True, exist_ok=True)
    valid_gt = np.isfinite(gt) & (gt > 0)
    valid = valid_pixels(pred, gt)
    scale = oracle_scale(pred, gt, valid)
    aligned = pred * scale if scale is not None else np.full(gt.shape, np.nan)
    overlay = rgb.astype(float).copy()
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([255, 0, 190])
    nan = np.full(gt.shape, np.nan)
    panels = [
        ('input', 'Transparent RGB input', rgb, None, None),
        ('gt', 'GT depth (m)', np.where(valid_gt, gt, np.nan), depth_max, 'm'),
        ('raw_metric', 'Raw metric depth (m)' if metric else 'Raw metric unavailable: no trusted K',
         np.where(valid_gt & np.isfinite(pred) & (pred > 0), pred, np.nan) if metric else nan, depth_max, 'm'),
        ('raw_error', 'Raw absolute error (m)' if metric else 'Raw error unavailable',
         np.where(valid, np.abs(pred-gt), np.nan) if metric else nan, error_max, 'm'),
        ('mask_overlay', 'Transparent mask overlay', overlay.astype(np.uint8), None, None),
        ('oracle_depth', 'Oracle GT-aligned depth (m): DIAGNOSTIC ONLY',
         np.where(valid, aligned, np.nan), depth_max, 'm'),
        ('oracle_error', 'Oracle residual (m): DIAGNOSTIC ONLY',
         np.where(valid, np.abs(aligned-gt), np.nan), error_max, 'm')]
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('#999999')
    def draw(ax, panel):
        _, title, data, maximum, unit = panel
        if maximum is None:
            ax.imshow(data)
        else:
            im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=maximum)
            ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=unit, extend='max')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    for panel in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        draw(ax, panel)
        fig.savefig(target / (panel[0]+'.png'), dpi=110, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    for ax, panel in zip(axes.flat, panels):
        draw(ax, panel)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis('off')
    fig.suptitle(f"{sample['split']}/{sample['sample_id']} | grey = invalid | fixed metre limits")
    fig.tight_layout()
    destination = target / 'comparison.png'
    fig.savefig(destination, dpi=110)
    plt.close(fig)
    return destination


def visualize_saved(output, data_root=None, depth_max=2., error_max=0.5):
    output = Path(output)
    manifest = read_json(output / 'sample_manifest.json')
    root = data_root or manifest['data_root']
    protect_dataset(root, output)
    thumbnails, failures = [], []
    for s in manifest['samples']:
        try:
            rgb = read_rgb(path_for(root, s, 'rgb'))
            gt, mask, _ = annotations(root, s, rgb.shape[:2])
            a, metric, _ = saved_prediction(output, s)
            file = render_sample(output, s, rgb, gt, mask, resize_prediction(a, gt.shape), metric,
                                 depth_max, error_max)
            with Image.open(file) as im:
                thumb = im.convert('RGB')
                thumb.thumbnail((480, 320))
            thumbnails.append(thumb)
        except Exception as exc:
            failures.append(dict(sample_id=s['sample_id'], split=s['split'], error=str(exc)))
    if thumbnails:
        columns = min(4, len(thumbnails))
        sheet = Image.new('RGB', (480*columns, 320*((len(thumbnails)+columns-1)//columns)), 'white')
        for i, thumb in enumerate(thumbnails):
            sheet.paste(thumb, (480*(i % columns), 320*(i//columns)))
        sheet.save(output / 'contact_sheet.png')
    write_json(output / 'visualization_summary.json', dict(rendered=len(thumbnails), failures=failures,
        depth_limits_m=[0, depth_max], error_limits_m=[0, error_max], auto_normalization=False))
    return failures
