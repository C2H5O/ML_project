"""Fixed metre colour limits across samples; grey invalid annotations."""
from pathlib import Path
import numpy as np
from PIL import Image
from src.common import read_json, write_json, protect_dataset
from src.datasets.cleargrasp import annotations, read_rgb, path_for
from src.evaluation.cleargrasp import saved_prediction
from src.evaluation.relative_geometry import oracle_scale
from src.evaluation.metrics import valid_pixels
from src.inference.da3_zero_shot import resize_prediction
from src.visualization.transcg import colour_depth
from src.datasets.cleargrasp_paths import cleargrasp_figure_dir, load_result_manifest


def depth_display_arrays(gt, pred):
    """GT validity applies to GT and errors, never to the predicted surface."""
    valid_gt = np.isfinite(gt) & (gt > 0)
    valid_pred = np.isfinite(pred) & (pred > 0)
    valid = valid_pixels(pred, gt)
    scale = oracle_scale(pred, gt, valid)
    aligned = pred * scale if scale is not None else np.full(gt.shape, np.nan)
    return dict(gt=np.where(valid_gt, gt, np.nan),
        raw_metric=np.where(valid_pred, pred, np.nan),
        raw_error=np.where(valid, np.abs(pred-gt), np.nan),
        oracle_depth=np.where(valid_pred, aligned, np.nan),
        oracle_error=np.where(valid, np.abs(aligned-gt), np.nan))


def render_sample(output, sample, rgb, gt, mask, pred, metric, depth_max=2., error_max=0.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if depth_max <= 0 or error_max <= 0:
        raise ValueError('Colourbar limits must be positive')
    target = cleargrasp_figure_dir(output, sample)
    target.mkdir(parents=True, exist_ok=True)
    display = depth_display_arrays(gt, pred)
    overlay = rgb.astype(float).copy()
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([255, 0, 190])
    nan = np.full(gt.shape, np.nan)
    panels = [
        ('input', 'Transparent RGB input', rgb, None, None),
        ('gt', 'GT depth (m)', display['gt'], depth_max, 'm'),
        ('raw_metric', 'Raw metric depth (m)' if metric else 'Raw metric unavailable: no trusted K',
         display['raw_metric'] if metric else nan, depth_max, 'm'),
        ('raw_error', 'Raw absolute error (m)' if metric else 'Raw error unavailable',
         display['raw_error'] if metric else nan, error_max, 'm'),
        ('mask_overlay', 'Transparent mask overlay', overlay.astype(np.uint8), None, None),
        ('oracle_depth', 'Oracle GT-aligned depth (m): DIAGNOSTIC ONLY',
         display['oracle_depth'], depth_max, 'm'),
        ('oracle_error', 'Oracle residual (m): DIAGNOSTIC ONLY',
         display['oracle_error'], error_max, 'm')]
    depth_cmap = plt.get_cmap('viridis').copy()
    error_cmap = plt.get_cmap('magma').copy()
    depth_cmap.set_bad('#999999')
    error_cmap.set_bad('#999999')
    def draw(ax, panel):
        _, title, data, maximum, unit = panel
        if maximum is None:
            ax.imshow(data)
        else:
            cmap = error_cmap if panel[0] in ('raw_error', 'oracle_error') else depth_cmap
            im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=maximum,
                           interpolation='nearest')
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
    fig.suptitle(f"{sample['split']}/{sample['sample_id']} | prediction shown independently of GT | fixed metre limits")
    fig.tight_layout()
    destination = target / 'comparison.png'
    fig.savefig(destination, dpi=110)
    plt.close(fig)
    write_json(target / 'visualization.json', dict(depth_colormap='viridis', error_colormap='magma',
        depth_limits_m=[0, depth_max], error_limits_m=[0, error_max], interpolation='nearest',
        prediction_mask='finite positive prediction only; independent of GT',
        gt_mask='finite positive GT only', error_mask='valid GT AND valid prediction',
        prediction_shape=list(pred.shape), gt_alignment_for_raw_metric=False,
        spatial_filters=False, auto_normalization=False))
    return destination


def visualize_saved(output, data_root=None, depth_max=2., error_max=0.5):
    output = Path(output)
    manifest = load_result_manifest(output)
    root = data_root or manifest['data_root']
    protect_dataset(root, output)
    thumbnails, failures = [], []
    for s in manifest['samples']:
        try:
            rgb = read_rgb(path_for(root, s, 'rgb'))
            gt, mask, _ = annotations(root, s, rgb.shape[:2])
            a, metric, _ = saved_prediction(output, s)
            resized = resize_prediction(a, gt.shape)
            file = render_sample(output, s, rgb, gt, mask, resized, metric,
                                 depth_max, error_max)
            if metric:
                # Full-frame exports reuse the established TransCG colouring.
                # Neither display is masked by GT or changes the saved prediction.
                colour_depth(a, 0, depth_max).save(file.parent / 'predicted_depth.png')
                colour_depth(resized, 0, depth_max).save(file.parent / 'predicted_depth_rgb_grid.png')
                settings_path = file.parent / 'visualization.json'
                settings = read_json(settings_path)
                v = np.isfinite(a) & (a > 0)
                settings.update(native_prediction_shape=list(a.shape),
                    predicted_depth_png='native network grid; no resize; fixed metre range; no GT mask',
                    predicted_depth_rgb_grid_png='bilinear prediction resize to RGB grid; fixed metre range; no GT mask',
                    native_pixels_above_colour_max=int(np.sum(v & (a > depth_max))),
                    native_valid_pixels=int(v.sum()))
                write_json(settings_path, settings)
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
        depth_limits_m=[0, depth_max], error_limits_m=[0, error_max], auto_normalization=False,
        depth_colormap='viridis', error_colormap='magma', prediction_masked_by_gt=False,
        prediction_pngs='native grid and RGB grid; fixed metre ranges; no GT masking or sharpening'))
    return failures
